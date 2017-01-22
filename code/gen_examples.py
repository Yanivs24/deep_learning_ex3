#!/usr/bin/python

import random

STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}

POS_SIZE = 500
NEG_SIZE = 500
TRAIN_SIZE = 10000
TEST_SIZE  = 10000
MAX_SEQ_SIZE = 30

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
	pos_examples = []
	neg_examples = []

	for _ in range(POS_SIZE):
		pos_examples.append(generate_example(True))	

	for _ in range(NEG_SIZE):
		neg_examples.append(generate_example(False))

	# write pos examples to file
	with open('pos_examples', 'w') as f:
		f.write('\n'.join(pos_examples))

	# write neg examples to file
	with open('neg_examples', 'w') as f:
		f.write('\n'.join(neg_examples))

	# generate train and test sets
	with open('train_set', 'w') as f:
		for _ in range(TRAIN_SIZE):
			is_pos = bool(random.randint(0,1))
			f.write("%s %s\n" % (generate_example(is_pos), 'pos' if is_pos else 'neg'))

	with open('test_set', 'w') as f:
		for _ in range(TEST_SIZE):
			is_pos = bool(random.randint(0,1))
			f.write("%s %s\n" % (generate_example(is_pos), 'pos' if is_pos else 'neg'))





