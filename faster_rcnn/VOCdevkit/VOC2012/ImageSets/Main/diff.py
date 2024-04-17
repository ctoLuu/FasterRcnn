with open("test.txt",'r') as file:
    list = file.readlines()
    for i in range(350):
        if ("test"+str(i)+"\n") in list:
            continue
        else:
            print(i)
