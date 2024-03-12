import os
import xml.etree.ElementTree as ET

# 指定XML文件夹路径
xml_folder = './Annotations'

# 遍历文件夹中的所有XML文件
for filename in os.listdir(xml_folder):
    if filename.endswith('.xml'):
        file_path = os.path.join(xml_folder, filename)
        
        # 解析XML文件
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 查找<object>元素
        object_elements = root.findall('object')
        
        # 如果没有<object>元素，则输出文件名
        if len(object_elements) == 0:
            print(filename)

