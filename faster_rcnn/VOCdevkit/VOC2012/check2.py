import os
import xml.etree.ElementTree as ET

# 指定包含XML文件的文件夹路径
folder_path = "./Annotations"

# 获取文件夹中所有XML文件的文件名
xml_files = [file for file in os.listdir(folder_path) if file.endswith('.xml')]

# 遍历每个XML文件
for xml_file in xml_files:
    xml_path = os.path.join(folder_path, xml_file)
    
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 查找<object>标签
    for obj in root.findall('object'):
        # 检查<object>标签的文本是否包含数字
        if any(char.isdigit() for char in obj.text):
            print(f"数字标签出现在文件 {xml_file}: {obj.text}")

