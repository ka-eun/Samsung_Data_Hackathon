import csv

def count_cities(file_train='./train_kor.csv'):
    f1 = open(file_train, 'r')
    r = csv.reader(f1)
    f2 = open('cities.csv', 'w', newline='')
    wr = csv.writer(f2)

    big_city = []
    small_city=[]

    for j, row in enumerate(r):
        if j!=0:
            for i, elem in enumerate(row):
                if i==7:
                    big_city.append(elem)

    big_city = list(set(big_city))

    dic = {}
    for i in range(len(big_city)):
        small_city.append([])
        dic[big_city[i]] = small_city[i]


    f1.close();
    f1 = open(file_train, 'r')
    r = csv.reader(f1)

    for j, row in enumerate(r):
        if j != 0:
            dic[row[7]].append(row[8])

    for i, elem in enumerate(big_city):
        dic[elem] = list(set(dic[elem]))
        dic[elem].insert(0, elem)
        wr.writerow(dic[elem])

    print(big_city)
    print(dic)

    f1.close()
    f2.close()


if __name__ == "__main__":
    count_cities()