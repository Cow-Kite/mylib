import torch

# graph.pt 파일 로드
file_path = './data/partitions/ogbn-products/2-parts/ogbn-products-partitions/part_1/graph.pt'
data = torch.load(file_path)

# 데이터 출력
print("Data:", data)

# 텍스트 파일로 변환
txt_file_path = 'graph.txt'
with open(txt_file_path, 'w') as f:
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            f.write(f"{key} (sampled):\n")
            # 텐서의 처음 10개와 마지막 10개 값 출력
            sampled_values = torch.cat([value[:10], value[-10:]])
            for val in sampled_values:
                f.write(f"{val.item()}\n")
            f.write(f"{key} size: {value.size()}\n")
        else:
            f.write(f"{key}: {value}\n")

print(f"Data has been saved to {txt_file_path}")
