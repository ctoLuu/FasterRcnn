import os
import xml.etree.ElementTree as ET

def convert_txt_to_xml(txt_file_path, xml_folder_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    root = ET.Element("annotations")

    for line in lines:
        label, x_min, y_min, x_max, y_max = line.strip().split()
        annotation = ET.SubElement(root, "annotation")
        ET.SubElement(annotation, "label").text = label
        ET.SubElement(annotation, "x_min").text = x_min
        ET.SubElement(annotation, "y_min").text = y_min
        ET.SubElement(annotation, "x_max").text = x_max
        ET.SubElement(annotation, "y_max").text = y_max

    tree = ET.ElementTree(root)
    xml_file_path = os.path.join(xml_folder_path, os.path.splitext(os.path.basename(txt_file_path))[0] + ".xml")
    tree.write(xml_file_path)

txt_file_path = "./label_txt"
xml_folder_path = "./Annotations2"

if not os.path.exists(xml_folder_path):
    os.makedirs(xml_folder_path)

convert_txt_to_xml(txt_file_path, xml_folder_path)

