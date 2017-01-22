#!/usr/bin/python

import random

STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}

POS_SIZE = 500
NEG_SIZE = 500
TRAIN_SIZE = 40000
TEST_SIZE  = 40000
MAX_SEQ_SIZE = 30



def generate_example2(is_positive):
    ''' More a's then b's '''

    digits = ['a','b']

    
    while True:
        sequence = []
        
        seq_size  = random.randint(50, 200)
        for _ in range(seq_size):
            sequence.append(random.choice(digits))

        more_a = sequence.count('a') >= sequence.count('b')
        if (more_a and is_positive) or (not more_a and not is_positive):
            return ''.join(sequence)


def generate_example3(is_positive):
    ''' More a's then b's '''

    digits = ['a','b']

    
    while True:
        sequence = []
        seq_size  = random.randint(50, 200)
        for _ in range(seq_size):
            sequence.append(random.choice(digits))

        got_pos = False
        for d in digits:
            got_pos |= (sequence[0] == d) and (sequence.count(d) >= 0.5 * len(sequence))

        if (got_pos and is_positive) or (not got_pos and not is_positive):
            return ''.join(sequence)


def generate_example4(is_positive):
    ''' More a's then b's '''

    digits = ['a','b']

    sequence = []
    seq_size  = random.randint(50, 100)
    for _ in range(seq_size):
        sequence.append(random.choice(digits))

    reverse_seq = sequence[:-1]
    reverse_seq.reverse()
    sequence += reverse_seq

    # change one char (opposite)
    if not is_positive:
        ind = random.randint(0, len(sequence)-1)
        sequence[ind] = 'a' if sequence[ind]=='b' else 'b'

    return ''.join(sequence)


def generate_example(is_positive):

    sequence = []
    digits = map(str, range(1, 10))
    if is_positive:
        letters = ['a','b','c','d']
    else:
        letters = ['a','c','b','d']

    # generate random sequence of digits followed by letters
    for l in letters:
        amount_digits  = random.randint(1, MAX_SEQ_SIZE)
        amount_letters = random.randint(1, MAX_SEQ_SIZE)
        for _ in xrange(amount_digits):
            sequence.append(random.choice(digits))
        for _ in xrange(amount_letters):
            sequence.append(l)

    # generate more random letters in the end
    amount_digits = random.randint(1, MAX_SEQ_SIZE)
    for _ in xrange(amount_digits):
        sequence.append(random.choice(digits))

    return ''.join(sequence)


if __name__ == "__main__":

    # generate train set
    with open('train_set', 'w') as f:
        for _ in range(TRAIN_SIZE):
            is_pos = bool(random.randint(0,1))
            f.write("%s %s\n" % (generate_example4(is_pos), 'pos' if is_pos else 'neg'))

    # generate test set
    with open('test_set', 'w') as f:
        for _ in range(TEST_SIZE):
            is_pos = bool(random.randint(0,1))
            f.write("%s %s\n" % (generate_example4(is_pos), 'pos' if is_pos else 'neg'))





