import random

winning_ticket = [7, 'b', 8, 'a']
tries = 0
while True:
    tries += 1
    my_ticket = [random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]) if i < 3 else random.choice(['a', 'b', 'c']) for i in
                 range(4)]
    if my_ticket == winning_ticket:
        print("We have a winning ticket!")
        print(f"Your ticket: {my_ticket}")
        print(f"Winning ticket: {winning_ticket}")
        print(f"It only took {tries} tries to win!")
        break