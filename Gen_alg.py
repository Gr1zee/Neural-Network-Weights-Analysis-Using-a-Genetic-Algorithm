import random


def generate_character(length_character):
    character = ""
    for _ in range(length_character):
        character += str(random.randint(0, 1))
    return character


def generate_list(length_character, length_list):
    list_characters = []
    for _ in range(length_list):
        character = generate_character(length_character)
        list_characters.append(character)
    return list_characters


def success_rate(list_characters):
    counter = 0
    for elem in list_characters:
        counter += elem.count("1")
    return counter / len(list_characters)


def counter(character):
    return character.count("1")


def best_character(list_character):
    best_c = "000000"
    c_best_character = 0
    for elem in list_character:
        if c_best_character < counter(elem):
            best_c = elem
            c_best_character = counter(elem)
    return best_c


def fight_club(list_characters):
    pers1 = random.choice(list_characters)
    pers2 = random.choice(list_characters)
    if counter(pers1) > counter(pers2):
        return pers1
    else:
        return pers2


def crossing(list_characters):
    offspring_list = []
    for i in range(len(list_characters)):
        gen_crossing = random.randint(0, len(list_characters[i]))
        child = fight_club(list_characters)[:gen_crossing] + fight_club(list_characters)[gen_crossing:]
        offspring_list.append(child)
    return offspring_list


def mutation(character, chance_mutation):
    mutated_character = ""
    for i in range(len(character)):
        mutation_check = random.randint(0, 1 // (chance_mutation / 100))
        if mutation_check == 1 and character[i] == "0":
            mutated_character += "1"
        elif mutation_check == 1 and character[i] == "1":
            mutated_character += "0"
        else:
            mutated_character += character[i]
    return character


def mutation_population(list_characters, chance_mutation):
    mutated_population = []
    for elem in list_characters:
        m = mutation(elem, chance_mutation)
        mutated_population.append(m)
    return mutated_population


def generation(current_population, chance_mutation):
    population = current_population
    parent_population = crossing(population)
    mutated_population = mutation_population(parent_population, chance_mutation)
    return mutated_population


def generate_population(number_of_generations, len_character, len_population, the_sought_character="",
                        chance_mutation=5):
    find_best_character = False
    if the_sought_character == "":
        the_sought_character = "1" * len_character
    generation_counter = 0
    if number_of_generations == 0:
        find_best_character = True
    while number_of_generations > 0 or find_best_character:
        first_population = generate_list(len_character, len_population)
        current_population = generation(first_population, chance_mutation)
        if find_best_character and the_sought_character in current_population:
            return current_population

        number_of_generations -= 1
        generation_counter += 1


if __name__ == '__main__':
    a = generate_population(0, 6, 20, "", 3)