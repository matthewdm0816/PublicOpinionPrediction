import pandas as pd

def xlsx_to_csv(xlsx_file, csv_file):
    """将XLSX文件转换为CSV文件"""
    # 读取Excel文件（默认第一个工作表）
    df = pd.read_excel(xlsx_file)
    
    # 保存为CSV文件（使用utf-8-sig编码确保Excel正确显示中文）
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ 转换成功: {xlsx_file} → {csv_file}")

# 使用示例
if __name__ == "__main__":
    xlsx_to_csv("data_new/0_300/微博话题.xlsx", "data_new/0_300/微博话题.csv")