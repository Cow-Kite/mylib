import argparse
import logging
import multiprocessing
import os
import queue
import re
import signal
import subprocess
import sys
import time
from functools import partial
from threading import Thread
from typing import Optional

# 원격 머신에서 실행 중인 프로세스 정리
def clean_runs(get_all_remote_pids, conn):
    """This process cleans up the remaining remote training tasks."""
    print("Cleanup runs")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    data = conn.recv()

    # If the launch process exits normally, don't do anything:
    if data == "exit":
        sys.exit(0)
    else:
        remote_pids = get_all_remote_pids()
        for (ip, port), pids in remote_pids.items():
            kill_proc(ip, port, pids)
    print("Cleanup exits")

# 지정된 IP와 port의 원격 머신에서 지정된 PID를 가진 프로세스 종료
def kill_proc(ip, port, pids):
    """SSH to remote nodes and kill the specified processes."""
    curr_pid = os.getpid()
    killed_pids = []
    pids.sort()
    for pid in pids:
        assert curr_pid != pid
        print(f"Kill process {pid} on {ip}:{port}", flush=True)
        kill_cmd = ("ssh -o StrictHostKeyChecking=no -p " + str(port) + " " +
                    ip + " 'kill {}'".format(pid))
        subprocess.run(kill_cmd, shell=True)
        killed_pids.append(pid)
    for i in range(3):
        killed_pids = get_pids_to_kill(ip, port, killed_pids)
        if len(killed_pids) == 0:
            break
        else:
            killed_pids.sort()
            for pid in killed_pids:
                print(f"Kill process {pid} on {ip}:{port}", flush=True)
                kill_cmd = ("ssh -o StrictHostKeyChecking=no -p " + str(port) +
                            " " + ip + " 'kill -9 {}'".format(pid))
                subprocess.run(kill_cmd, shell=True)

# 지정된 원격 머신에서 아직 살아있는 프로세스 ID를 가져옴
def get_pids_to_kill(ip, port, killed_pids):
    """Get the process IDs that we want to kill but are still alive."""
    killed_pids = [str(pid) for pid in killed_pids]
    killed_pids = ",".join(killed_pids)
    ps_cmd = ("ssh -o StrictHostKeyChecking=no -p " + str(port) + " " + ip +
              " 'ps -p {} -h'".format(killed_pids)) # ps 명령으로 프로세스 상태를 확인
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids = []
    for p in res.stdout.decode("utf-8").split("\n"):
        ps = p.split()
        if len(ps) > 0:
            pids.append(int(ps[0]))
    return pids

# SSH를 통해 원격 머신에서 명령을 실행하는 기능 제공
# 명령을 실행하기 위해 스레드를 생성하고, 실행 결과를 큐에 넣음
def remote_execute(
    cmd: str, # 원격 머신에서 실행할 사용자 정의 명령
    state_q: queue.Queue, # 스레드 상태를 수집하는 큐
    ip: str, # 명령을 실행할 호스트의 IP
    port: int, # 호스트가 리스닝 중인 port 번호
    username: Optional[str] = None,
) -> Thread: # 명령이 완료되면 반환
    """Execute command line on remote machine via ssh.

    Args:
        cmd: User-defined command (udf) to execute on the remote host.
        state_q: A queue collecting Thread exit states.
        ip: The ip-address of the host to run the command on.
        port: Port number that the host is listening on.
        username: If given, this will specify a username to use when issuing
            commands over SSH. Useful when your infra requires you to
            explicitly specify a username to avoid permission issues.

    Returns:
        thread: The thread who runs the command on the remote host.
            Returns when the command completes on the remote host.
    """
    ip_prefix = ""
    if username is not None:
        ip_prefix += f"{username}@"

    # Construct ssh command that executes `cmd` on the remote host
    ssh_cmd = (f"ssh -o StrictHostKeyChecking=no -p {port} {ip_prefix}{ip} "
               f"'{cmd}'")

    print(f"----- ssh_cmd={ssh_cmd} ")

    # thread func to run the job
    def run(ssh_cmd, state_q):
        try:
            subprocess.check_call(ssh_cmd, shell=True) # ssh_cmd 실행
            state_q.put(0) # 실행 결과를 state_q에 넣음
        except subprocess.CalledProcessError as err:
            print(f"Called process error {err}")
            state_q.put(err.returncode)
        except Exception:
            state_q.put(-1)

    # run 함수를 실행하는 스레드를 생성하고 시작함
    thread = Thread(
        target=run,
        args=(
            ssh_cmd,
            state_q,
        ),
    )
    thread.setDaemon(True)
    thread.start()
    # Sleep for a while in case SSH is rejected by peer due to busy connection:
    time.sleep(0.2) # SSH 연결 대기
    return thread # 스레드 반환

# 원격 머신에서 특정 명령을 실행 중인 프로세스 ID(PID)를 가져오는 역할을 함 (단일 머신)
# 이 함수는 SSH를 통해 원격 머신에 접속하여 ps와 pgrep 명령어를 사용하여 해당 명령을 실행 중인 프로세스를 찾아냄
# 분산 시스템에서 원격 머신에서 실행 중인 특정 명령과 관련된 모든 프로세스를 식별하는 데 유용함
# 이를 통해 원격 머신에서 실행 중인 작업을 모니터링하고 관리할 수 있음
def get_remote_pids(ip, port, cmd_regex): # ip: 원격 호스트 IP 주소, port: 원격 호스트가 리스닝 중인 포트 번호, cmd_regex: 특정 명령을 식별하기 위한 정규 표현식
    """Get the process IDs that run the command in the remote machine."""
    pids = [] # 원격 호스트에서 실행 중인 특정 명령과 일치하는 모든 프로세스 ID를 정렬된 리스트로 반환
    curr_pid = os.getpid() # 현재 프로세스 ID를 가져옴
    # We want to get the Python processes. However, we may get some SSH
    # processes, so we should filter them out:
    # ssh를 통해 원격 머신에서 python 프로세스를 찾음
    # 여기서 grep -v StrictHostKeyChecking은 SSH 프로세스를 필터링함
    ps_cmd = (f"ssh -o StrictHostKeyChecking=no -p {port} {ip} "
              f"'ps -aux | grep python | grep -v StrictHostKeyChecking'")
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    for p in res.stdout.decode("utf-8").split("\n"):
        ps = p.split() 
        if len(ps) < 2:
            continue
        # We only get the processes that run the specified command:
        res = re.search(cmd_regex, p)
        if res is not None and int(ps[1]) != curr_pid:
            pids.append(ps[1])

    pid_str = ",".join([str(pid) for pid in pids])
    ps_cmd = (f"ssh -o StrictHostKeyChecking=no -p {port} {ip} "
              f" 'pgrep -P {pid_str}'")
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids1 = res.stdout.decode("utf-8").split("\n")
    all_pids = []
    for pid in set(pids + pids1):
        if pid == "" or int(pid) == curr_pid:
            continue
        all_pids.append(int(pid))
    all_pids.sort()
    return all_pids

# 여러 원격 호스트에서 특정 명령을 실행 중인 모든 프로세스 ID를 가져오는 역할 (여러 머신)
def get_all_remote_pids(hosts, ssh_port, udf_command):
    """Get all remote processes."""
    remote_pids = {}
    for node_id, host in enumerate(hosts):
        ip, _ = host
        # When creating training processes in remote machines, we may insert
        # some arguments in the commands. We need to use regular expressions to
        # match the modified command.
        cmds = udf_command.split()
        new_udf_command = " .*".join(cmds)
        pids = get_remote_pids(ip, ssh_port, new_udf_command)
        remote_pids[(ip, ssh_port)] = pids
    return remote_pids

# 주어진 명령어에 환경 변수를 설정하는 쉘 명령을 추가하여 매핑하는 역할
def wrap_cmd_w_envvars(cmd: str, env_vars: str) -> str:
    """Wraps a CLI command with desired environment variables.

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = "VAR1=value1 VAR2=value2"
        >>> wrap_cmd_w_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"
    """
    if env_vars == "":
        return f"({cmd})"
    else:
        return f"(export {env_vars}; {cmd})"
    
# 주어진 명령어에 추가적인 환경변수를 설정하는 쉘 명령을 추가
def wrap_cmd_w_extra_envvars(cmd: str, env_vars: list) -> str:
    """Wraps a CLI command with extra environment variables.

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = ["VAR1=value1", "VAR2=value2"]
        >>> wrap_cmd_w_extra_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"
    """
    env_vars = " ".join(env_vars)
    return wrap_cmd_w_envvars(cmd, env_vars)

def get_available_port(ip):
    """Get available port with specified ip."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for port in range(1234, 65535):
        try:
            sock.connect((ip, port))
        except Exception:
            return port
    raise RuntimeError(f"Failed to get available port for ip~{ip}")

def submit_all_jobs(args, udf_command, dry_run=False):
    if dry_run:
        print("Dry run mode, no jobs will be launched")

    servers_cmd = []
    hosts = []
    thread_list = []

    ip_config = os.path.join(args.workspace, args.ip_config)
    with open(ip_config) as f:
        for line in f:
            result = line.strip().split()
            if len(result) == 2:
                ip = result[0]
                port = int(result[1])
                hosts.append((ip, port))
            elif len(result) == 1: # port 정보가 없으면 get_available_port 함수를 사용하여 포트를 할당
                ip = result[0]
                port = get_available_port(ip)
                hosts.append((ip, port))
            else:
                raise RuntimeError("Format error of 'ip_config'")
            
    state_q = queue.Queue()

    # 마스터 서버의 ip를 설정하고 각 서버에서 실행할 명령어를 생성함
    # 각 서버에 대해 wrap_cmd_w_envvars와 wrap_cmd_w_extra_envvars함수를 사용하여 환경 변수를 추가하고 명령어를 생성함
    # dry_run이 아닌 경우, remote_execute 함수를 사용하여 원격 서버에서 명령을 실행하는 스레드를 생성함
    master_ip, _ = hosts[0]
    for i in range(len(hosts)):
        ip, _ = hosts[i]
        server_env_vars_cur = ""
        cmd = wrap_cmd_w_envvars(udf_command, server_env_vars_cur)
        cmd = (wrap_cmd_w_extra_envvars(cmd, args.extra_envs)
               if len(args.extra_envs) > 0 else cmd)
        
        cmd = cmd[:-1]
        cmd += f" --dataset_root_dir={args.dataset_root_dir}"
        cmd += f" --dataset={args.dataset}"
        cmd += f" --num_nodes={args.num_nodes}"
        cmd += f" --node_rank={i}"
        cmd += f" --master_addr={master_ip}"
        servers_cmd.append(cmd)

        if not dry_run:
            thread_list.append(
                remote_execute(cmd, state_q, ip, args.ssh_port,
                               username=args.ssh_username))

    # 파이프 및 프로세스 생성
    # 파이프를 생성하고, 원격 서버의 PID를 가져오는 get_all_remote_pids 함수를 실행할 프로세스를 생성함
    conn1, conn2 = multiprocessing.Pipe()
    func = partial(get_all_remote_pids, hosts, args.ssh_port, udf_command)
    process = multiprocessing.Process(target=clean_runs, args=(func, conn1))
    process.start()

    def signal_handler(signal, frame):
        logging.info("Stop launcher")
        conn2.send("cleanup")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 스레드 종료 및 상태 확인
    # 모든 스레드가 종료될 때까지 기다리고, 각 스레드의 상태 코드를 확인함
    err = 0
    for thread in thread_list:
        thread.join()
        err_code = state_q.get()
        if err_code != 0:
            err = err_code

    # 프로세스 종료 및 상태 출력
    conn2.send("exit")
    process.join()
    if err != 0:
        print("Task failed")
        sys.exit(-1)
    print("=== fully done ! === ")

def main():
    parser = argparse.ArgumentParser(description="Launch a distributed job")
    parser.add_argument(
        "--ssh_port",
        type=int,
        default=22,
        help="SSH port",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Path of user directory of distributed tasks",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default='./data/products',
        help="The root directory (relative path) of partitioned dataset",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=2,
        help="Number of distributed nodes",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The current node rank",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default='localhost',
        help="The master address"
    )
    parser.add_argument(
        "--ip_config",
        required=True,
        type=str,
        help="File (in workspace) of IP configuration for server processes",
    )
    args, udf_command = parser.parse_known_args()

    udf_command = str(udf_command[0])
    if "python3" not in udf_command:
        raise RuntimeError("Launching script does only support a Python "
                           "executable file")
    submit_all_jobs(args, udf_command)

if __name__ == "__main__":
    main()

