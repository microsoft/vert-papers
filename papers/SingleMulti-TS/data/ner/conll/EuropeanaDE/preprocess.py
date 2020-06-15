def main():
    fns = ['enp_DE.lft.bio.txt', 'enp_DE.sbb.bio.txt']

    marks = ['.', '!', '?']
    
    res = []
    sent = []
    n_sents = 0

    # read
    for fn in fns:
        print('Processing {}...'.format(fn))
        with open (fn, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr):
                line = line.strip()
                if len(line.split()) != 2:
                    print('Invalid line: #{}'.format(idx))
                    continue
                
                sent.append(line.split()[0] + ' ' + 'O') # unused label, set to 'O'
                if line[0] in marks:
                    if len(sent) > 1: # single sentence of `.`
                        res.append(sent)
                        sent = []
                        n_sents += 1
        print('Current # sentences: {}'.format(n_sents))

    # write
    with open('train.txt', 'w', encoding='utf-8') as fw:
        for sent in res:
            fw.write('\n'.join(sent) + '\n\n')

if __name__ == '__main__':
    main()