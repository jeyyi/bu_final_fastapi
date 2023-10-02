def generate_tgwords():
    with open("app/resources/tagalog_stop_words.txt") as f:
        words = [line.strip() for line in f.readlines()]
    return words