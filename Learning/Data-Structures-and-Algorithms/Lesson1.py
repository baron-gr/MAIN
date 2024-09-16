### Binary Search, Linked Lists and Complexity ###

## Card Search ##

## Initial function signature
def locate_card(cards, query):
    pass

## Test case
cards = [13,11,10,7,4,3,1,0]
query = 7
output = 3

result = locate_card(cards, query)
# print(result)

# print(result == output)

## Test case dictionary
test = {
    'test_case' : 0,
    'input': {
        'cards': [13,11,10,7,4,3,1,0],
        'query': 7
    },
    'output': 3
}

# print(locate_card(**test['input']) == test['output'])

## Test cases
tests = []

## 1. Query occurs in the middle
tests.append(test)

tests.append({
    'test_case' : 1,
    'input': {
        'cards': [13,11,10,7,4,3,1,0],
        'query': 1
    },
    'output': 6
})

## 2. Query is the first element
tests.append({
    'test_case' : 2,
    'input': {
        'cards': [4,2,1,-1],
        'query': 4
    },
    'output': 0
})

## 3. Query is the last element
tests.append({
    'test_case' : 3,
    'input': {
        'cards': [3,-1,-9,-127],
        'query': -127
    },
    'output': 3
})

## 4. Cards contain only one element
tests.append({
    'test_case' : 4,
    'input': {
        'cards': [6],
        'query': 6
    },
    'output': 0
})

## 5. If cards does not contain query -> -1
tests.append({
    'test_case' : 5,
    'input': {
        'cards': [13,11,10,7,3,1,0],
        'query': 4
    },
    'output': -1
})

## 6. Cards is empty
tests.append({
    'test_case' : 6,
    'input': {
        'cards': [],
        'query': 7
    },
    'output': -1
})

## 7. Numbers repeat
tests.append({
    'test_case' : 7,
    'input': {
        'cards': [8,8,6,6,6,6,3,2,2,2,0,0,0],
        'query': 3
    },
    'output': 6
})

## 8. Query occurs multiple times -> first instance
tests.append({
    'test_case' : 8,
    'input': {
        'cards': [8,8,6,6,6,6,6,6,3,2,2,2,0,0,0],
        'query': 6
    },
    'output': 2
})

## Linear search
def locate_card(cards, query):
    # Create a variable position with the value 0
    position = 0
    # Set up loop for repetition
    while True:
        # Check if current element position matches query
        if cards[position] == query:
            # Answer found! Return and exit..
            return position
        # Increment position
        position += 1
        # Check if end of array
        if position == len(cards):
            # Number not found, return -1
            return -1

# print(test)
# print(locate_card(**test['input']) == test['output'])

# for test in tests:
#     print(test['test_case'])
#     print(locate_card(**test['input']) == test['output'])

## Logging
def locate_card(cards, query):
    position = 0
    
    print('cards:', cards)
    print('query:', query)
    
    while True:
        print('position:', position)
        
        if cards[position] == query:
            return position
        position += 1
        if position == len(cards):
            return -1

cards6 = tests[6]['input']['cards']
query6 = tests[6]['input']['query']
# print(locate_card(cards6, query6))

## Fix issue where list index out of range
def locate_card(cards, query):
    position = 0
    while position < len(cards):
        if cards[position] == query:
            return position
        position += 1
    return -1

# print(locate_card(cards6, query6))
# for test in tests:
#     print(test['test_case'])
#     print(locate_card(**test['input']) == test['output'])

## Complexity: Time complexity & space (no. variables) complexity
## e.g. cN^3 + dN^2 + eN + f -> O(N^3)
## Linear Search Complexity: Time = O(N), Space = O(1)

## Binary search
def locate_card(cards, query):
    lo = 0
    hi = len(cards) - 1
    count = 1
    
    while lo <= hi:
        mid = (lo + hi) // 2 # // returns quotient
        mid_number = cards[mid]
        
        print("Iter:",count,", lo:",lo,", hi:",hi,", mid:",mid,", mid_number:",mid_number)
        
        if mid_number == query:
            return mid
        elif mid_number < query:
            hi = mid - 1
        elif mid_number > query:
            lo = mid + 1
        count += 1
    return -1

# for test in tests:
#     print("\nTest Case:",test['test_case'])
#     print(test['input'])
#     print("Actual Output:",locate_card(**test['input']))
#     print("Expected Output:",test['output'])

## Fix issue when cards[mid] == query: check whether it is first occurrence
def test_location(cards, query, mid):
    mid_number = cards[mid]
    if mid_number == query:
        if mid-1 >= 0 and cards[mid-1] == query:
            return 'left'
        else:
            return 'found'
    elif mid_number < query:
        return 'left'
    else:
        return 'right'

def locate_card(cards, query):
    lo, hi = 0, len(cards) - 1
    count = 1
    
    while lo <= hi:
        print("Iter:",count,", lo:",lo,", hi:",hi)
        mid = (lo + hi) // 2 # // returns quotient
        result = test_location(cards, query, mid)    
        
        if result == 'found':
            return mid
        elif result == 'left':
            hi = mid - 1
        elif result == 'right':
            lo = mid + 1
        count += 1
    return -1

# for test in tests:
#     print("\nTest Case:",test['test_case'])
#     print(test['input'])
#     print("Actual Output:",locate_card(**test['input']))
#     print("Expected Output:",test['output'])

## Iteration k, N/(2^k), final length of array is 1, therefore N/(2^k) = 1
## k = log_2(N)
## Binary Seach Complexity: Time = O(log N), Space = O(1)

## Binary vs Linear Search
def locate_card_linear(cards, query):
    position = 0
    while position < len(cards):
        if cards[position] == query:
            return position
        position += 1
    return -1

def test_location(cards, query, mid):
    mid_number = cards[mid]
    if mid_number == query:
        if mid-1 >= 0 and cards[mid-1] == query:
            return 'left'
        else:
            return 'found'
    elif mid_number < query:
        return 'left'
    else:
        return 'right'

def locate_card_binary(cards, query):
    lo, hi = 0, len(cards) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        result = test_location(cards, query, mid)    
        if result == 'found':
            return mid
        elif result == 'left':
            hi = mid - 1
        elif result == 'right':
            lo = mid + 1
    return -1

large_test = {
    'test_case': 9,
    'input': {
        'cards': list(range(10000000,0,-1)),
        'query': 2
    },
    'output': 9999998
}

# print(locate_card_linear(**large_test['input']))
# print(locate_card_binary(**large_test['input']))

## Generic Binary Search Algorithm
def binary_search(lo, hi, condition):
    """ 
    lo: minimum list index
    hi: maximum list index
    condition: function that returns found, left, right
    """
    while lo <= hi:
        mid = (lo + hi) // 2
        result = condition(mid)
        if result == 'found':
            return mid
        elif result == 'left':
            hi = mid - 1
        else:
            lo = mid + 1
    return -1

## Func in a func
def locate_card(cards, query):
    def condition(mid):
        if cards[mid] == query:
            if mid > 0 and cards[mid-1] == query:
                return 'left'
            else:
                return 'found'
        elif cards[mid] < query:
            return 'left'
        else:
            return 'right'
    return binary_search(0, len(cards) - 1, condition)

# for test in tests:
#     print("\nTest Case:",test['test_case'])
#     print(test['input'])
#     print("Actual Output:",locate_card(**test['input']))
#     print("Expected Output:",test['output'])

## Cards ascending, find first and last index of given number
def first_position(nums, target):
    def condition(mid):
        if nums[mid] == target:
            if mid > 0 and nums[mid-1] == target:
                return 'left'
            else:
                return 'found'
        elif nums[mid] < target:
            return 'right' # Now right because in ascending order
        else:
            return 'left' # Now left because in ascending order
    return binary_search(0, len(nums) - 1, condition)

def last_position(nums, target):
    def condition(mid):
        if nums[mid] == target:
            if mid < len(nums) - 1 and nums[mid+1] == target:
                return 'right'
            else:
                return 'found'
        elif nums[mid] < target:
            return 'right'
        else:
            return 'left'
    return binary_search(0, len(nums) - 1, condition)

def first_and_last_position(nums, target):
    return first_position(nums, target), last_position(nums, target)

first_last_list = {
    'input': {
        'nums': [0,0,2,3,4,4,6,6,6,6,7,7,8,8,8],
        'target': 6
    },
    'output': (6, 9)
}

# print(first_and_last_position(**first_last_list['input'])==first_last_list['output'])