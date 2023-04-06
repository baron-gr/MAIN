import re

def get_pattern_match(pattern, text):
    matches = re.findall(pattern, text)
    if matches:
        return matches[0]

chat1 = 'codebasics: you ask lot of questions 😠  1235678912, abc@xyz.com'
chat2 = 'codebasics: here it is: (123)-567-8912, abc@xyz.com'
chat3 = 'codebasics: yes, phone: 1235678912 email: abc@xyz.com'

## Phone number matches
phone1 = get_pattern_match('(\d{10})|(\(\d{3}\)-\d{3}-\d{4})',chat1)
phone2 = get_pattern_match('(\d{10})|(\(\d{3}\)-\d{3}-\d{4})',chat2)
phone3 = get_pattern_match('(\d{10})|(\(\d{3}\)-\d{3}-\d{4})',chat3)
# print(phone1,phone2,phone3)

## Email matches
email1 = get_pattern_match('[a-zA-Z0-9_]*@[a-zA-Z0-9]*\.[a-zA-Z]*',chat1)
email2 = get_pattern_match('[a-zA-Z0-9_]*@[a-zA-Z0-9]*\.[a-zA-Z]*',chat2)
email3 = get_pattern_match('[a-zA-Z0-9_]*@[a-zA-Z0-9]*\.[a-zA-Z]*',chat3)
# print(email1,email2,email3)

chat4 = 'codebasics: Hello, I am having an issue with my order # 412889912'
chat5 = 'codebasics: I have a problem with my order number 412889912'
chat6 = 'codebasics: My order 412889912 is having an issue, I was charged 300$ when online it says 280$'

## Order matches
order4 = get_pattern_match('order[^\d]*(\d*)',chat4)
order5 = get_pattern_match('order[^\d]*(\d*)',chat5)
order6 = get_pattern_match('order[^\d]*(\d*)',chat6)
# print(order4,order5,order6)

## Regex for information extraction
text='''
Born	Elon Reeve Musk
June 28, 1971 (age 50)
Pretoria, Transvaal, South Africa
Citizenship	
South Africa (1971-present)
Canada (1971-present)
United States (2002-present)
Education	University of Pennsylvania (BS, BA)
Title	
Founder, CEO and Chief Engineer of SpaceX
CEO and product architect of Tesla, Inc.
Founder of The Boring Company and X.com (now part of PayPal)
Co-founder of Neuralink, OpenAI, and Zip2
Spouse(s)	
Justine Wilson

(m. 2000; div. 2008)
Talulah Riley
(m. 2010; div. 2012)
(m. 2013; div. 2016)
'''

## Age
age = get_pattern_match('age (\d+)',text)
# print(age)

## Name
name = get_pattern_match('Born(.*)',text).strip()
# print(name)

## DoB
DoB = get_pattern_match('Born.*\n(.*)\(age',text).strip()
# print(DoB)

## Birth Place
place = get_pattern_match('\(age.*\n(.*)',text).strip()
# print(place)

## Total function
def get_personal_info(text):
    age = get_pattern_match('age (\d+)',text)
    name = get_pattern_match('Born(.*)',text).strip()
    DoB = get_pattern_match('Born.*\n(.*)\(age',text).strip()
    place = get_pattern_match('\(age.*\n(.*)',text).strip()
    
    return{
        'Age':int(age),
        'Full Name':name,
        'Date of Birth':DoB,
        'Place of Birth':place
    }

print(get_personal_info(text))