filename = input("Enter a filename: ")
vowels = {'A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u'}
vowel_count = 0
consonant_count = 0

with open(filename, 'r') as file:
    for char in file.read():
        if char.isalpha():
            if char in vowels:
                vowel_count += 1
            else:
                consonant_count += 1

print(f"The number of vowels is {vowel_count} and consonants is {consonant_count}")
