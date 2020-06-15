def main():
    fns = ['train_origin.txt']

    res = []
    sent = []
    n_sent = 0
    for fn in fns:
        print('Processing {}...'.format(fn))
        with open(fn, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr):
                line = line.strip()
                if len(line) == 0:
                    if len(sent) <=0:
                        raise ValueError('Invalid line #{}.'.format(idx))
                    res.append(sent)
                    sent = []
                    n_sent += 1
                    continue

                line = line.split()
                if len(line) != 5:
                    raise ValueError('Invalid line #{}.'.format(idx))

                sent.append(line[0] + ' ' + 'O') # unused label, set to 'O'
        print('Current # sentences: {}'.format(n_sent))

    with open('train.txt', 'w', encoding='utf-8') as fw:
        for sent in res:
            fw.write('\n'.join(sent) + '\n\n')


if __name__ == '__main__':
    main()