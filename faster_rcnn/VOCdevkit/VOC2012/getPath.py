with open('./ImageSets/Main/test.txt') as file:
    test_img_names = [line.strip() for line in file]

with open('./ImageSets/Main/testName.txt','w') as file:
    for line in test_img_names:
        file.write(line+'.jpg'+'\n')