import os
import xml.etree.ElementTree as ET

# 源文件夹路径和目标文件夹路径
source_folder = './Annotations'
target_folder = './Annotations2'

# 创建目标文件夹
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹内的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.xml'):
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, filename)
        
        # 解析xml文件
        tree = ET.parse(source_file)
        root = tree.getroot()
        
        # 修改<filename>标签中的内容为文件名
        for elem in root.iter('filename'):
            elem.text = os.path.splitext(filename)[0]
        
        # 保存修改后的xml文件到目标文件夹
        tree.write(target_file)

print("XML文件中<filename>标签内容修改并保存到目标文件夹完成！")

