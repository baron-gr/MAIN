from typing import List

def sort_words(words: List[str]) -> List[str]:
    new_list = []
    for i in range(0,len(words)):
        for j in range(i+1, len(words)):
            if words[i] > words[j]:
                words[i], words[j] = words[j], words[i]
    new_list.append(words)
    return new_list

def sort_numbers(numbers: List[int]) -> List[int]:
    new_list = []
    for i in range(0, len(numbers)):
        for j in range(i+1, len(numbers)):
            if numbers[i] >= numbers[j]:
                numbers[i], numbers[j] = numbers[j], numbers[i]
    new_list.append(numbers)
    return new_list

def sort_decimals(numbers: List[float]) -> List[float]:
    new_list = []
    for i in range(0, len(numbers)):
        for j in range(i+1, len(numbers)):
            if numbers[i] >= numbers[j]:
                numbers[i], numbers[j] = numbers[j], numbers[i]
    new_list.append(numbers)
    return new_list


print(sort_words(["cherry", "apple", "blueberry", "banana", "watermelon", "zucchini", "kiwi", "pear"]))
# print(sort_numbers([1, 5, 3, 2, 4, 11, 19, 9, 2, 5, 6, 7, 4, 2, 6]))
# print(sort_decimals([3.14, 2.82, 6.433, 7.9, 21.555, 21.554]))
