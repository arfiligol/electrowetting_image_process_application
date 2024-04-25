current_index = 0
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(len(array))
while (True):
    if current_index < len(array):
        print("Not done yet.")
        print(current_index)
        current_index += 1
    else:
        break