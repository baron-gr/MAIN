## Write function to determine min number of rotations applied to list to return to sorted state

## 'nums': sorted rotated list e.g. [2,3,0,1]
## 'rotations': number of rotations applied to list e.g. 2

def count_rotations(nums):
    pass

tests = []

## Case 0
test = {
    'test_case': 0,
    'input': {
        'nums': [19,25,29,3,5,6,7,9,11,14]
    },
    'output': 3
}

tests.append(test)

## Case 1: List of size 8 rotated 5 times
tests.append({
    'test_case': 1,
    'input': {
        'nums': [4,5,6,7,8,1,2,3]
    },
    'output': 5
})

## Case 2: List not rotated
tests.append({
    'test_case': 2,
    'input': {
        'nums': [1,2,3]
    },
    'output': 0
})

## Case 3: List rotated once
tests.append({
    'test_case': 3,
    'input': {
        'nums': [4,1,2,3]
    },
    'output': 1
})

## Case 4: List rotated n-1 times
tests.append({
    'test_case': 4,
    'input': {
        'nums': [2,3,4,5,1]
    },
    'output': 4
})

## Case 5: List rotated n times
tests.append({
    'test_case': 5,
    'input': {
        'nums': [1,2,3,4,5,6]
    },
    'output': 0
})

## Case 6: Empty list
tests.append({
    'test_case': 6,
    'input': {
        'nums': []
    },
    'output': 0
})

## Case 7: List containing one element
tests.append({
    'test_case': 7,
    'input': {
        'nums': [1]
    },
    'output': 0
})

# for test in tests:
#     print("\nTest Case:", test['test_case'])
#     print("Rotated Sorted List:", test['input']['nums'])
#     print("Expected Output:", test['output'])
#     print("Actual Output", count_rotations(test['input']['nums']))
#     print("T/F:", count_rotations(test['input']['nums'])==test['output'])

## List rotated k times -> smallest value ends up in position k

## Own attempt
def count_rotations(list):
    if len(list) <= 1:
        return -1
    elif list.index(min(list)) == 0:
        return 0
    else:
        return list.index(min(list))

## Linear search method
def count_rotations_linear(nums):
    position = 0
    while position < len(nums):
        if position > 0 and nums[position] < nums[position-1]:
            return position
        position += 1
    return 0

## Binary search method
def count_rotations_binary(nums):
    lo = 0
    hi = len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        mid_number = nums[mid]
        if mid > 0 and mid_number == min(nums):
            return mid
        elif mid > min(nums):
            hi = mid - 1
        else:
            lo = mid + 1 
    return 0

for test in tests:
    print("\nTest Case:", test['test_case'])
    print("Rotated Sorted List:", test['input']['nums'])
    print("Expected Output:", test['output'])
    print("Actual Output", count_rotations_binary(test['input']['nums']))
    print("T/F:", count_rotations_binary(test['input']['nums'])==test['output'])