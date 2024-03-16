from .fewshotsampler import FewshotSampleBase

def get_class_name(rawtag):
    if rawtag.startswith('B-') or rawtag.startswith('I-'):
        return rawtag[2:]
    else:
        return rawtag

def convert_bio2spans(tags, schema):
    spans = []
    cur_span = []
    err_cnt = 0
    for i in range(len(tags)):
        if schema == 'BIO':
            if tags[i].startswith("B-") or tags[i] == 'O':
                if len(cur_span):
                    cur_span.append(i - 1)
                    spans.append(cur_span)
                    cur_span = []
            if tags[i].startswith("B-"):
                cur_span.append(tags[i][2:])
                cur_span.append(i)
            elif tags[i].startswith("I-"):
                if len(cur_span) == 0:
                    cur_span = [tags[i][2:], i]
                    err_cnt += 1
                if cur_span[0] != tags[i][2:]:
                    cur_span.append(i - 1)
                    spans.append(cur_span)
                    cur_span = [tags[i][2:], i]
                    err_cnt += 1
                assert cur_span[0] == tags[i][2:]
            else:
                assert tags[i] == "O"
        elif schema == 'IO':
            if tags[i] == "O":
                if len(cur_span):
                    cur_span.append(i - 1)
                    spans.append(cur_span)
                    cur_span = []
            elif (i == 0) or (tags[i] != tags[i - 1]):
                if len(cur_span):
                    cur_span.append(i - 1)
                    spans.append(cur_span)
                    cur_span = []
                cur_span.append(tags[i].strip("I-"))
                cur_span.append(i)
            else:
                assert cur_span[0] == tags[i].strip("I-")
        elif schema == "BIOES":
            if tags[i] == "O":
                if len(cur_span):
                    cur_span.append(i - 1)
                    spans.append(cur_span)
                    cur_span = []
            elif tags[i][0] == "S":
                if len(cur_span):
                    cur_span.append(i - 1)
                    spans.append(cur_span)
                    cur_span = []
                spans.append([tags[i][2:], i, i])
            elif tags[i][0] == "E":
                if len(cur_span) == 0:
                    err_cnt += 1
                    continue
                cur_span.append(i)
                spans.append(cur_span)
                cur_span = []
            elif tags[i][0] == "B":
                if len(cur_span):
                    cur_span.append(i - 1)
                    spans.append(cur_span)
                    cur_span = []
                cur_span = [tags[i][2:], i]
            else:
                if len(cur_span) == 0:
                    cur_span = [tags[i][2:], i]
                    err_cnt += 1
                    continue
                if cur_span[0] != tags[i][2:]:
                    cur_span.append(i - 1)
                    spans.append(cur_span)
                    cur_span = [tags[i][2:], i]
                    err_cnt += 1
                    continue
        else:
            raise ValueError
    if len(cur_span):
        cur_span.append(len(tags) - 1)
        spans.append(cur_span)
    return spans

class SpanSample(FewshotSampleBase):
    def __init__(self, idx, filelines, bio=True):
        super(SpanSample, self).__init__()
        self.index = idx
        filelines = [line.split('\t') for line in filelines]
        if len(filelines[0]) == 2:
            self.words, self.tags = zip(*filelines)
        else:
            self.words, self.postags, self.tags = zip(*filelines)
        self.spans = convert_bio2spans(self.tags, "BIO" if bio else "IO")
        return

    def get_max_ent_len(self):
        max_len = -1
        for sp in self.spans:
            max_len = max(max_len, sp[2] - sp[1] + 1)
        return max_len

    def get_num_of_long_ent(self, max_span_len):
        cnt = 0
        for sp in self.spans:
            if sp[2] - sp[1] + 1 > max_span_len:
                cnt += 1
        return cnt

    def __count_entities__(self):
        self.class_count = {}
        for tag, i, j in self.spans:
            if tag in self.class_count:
                self.class_count[tag] += 1
            else:
                self.class_count[tag] = 1
        return

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self):
        tag_class = list(set(map(lambda x: x[2:] if x[:2] in ['B-', 'I-', 'E-', 'S-'] else x, self.tags)))
        if 'O' in tag_class:
            tag_class.remove('O')
        return tag_class

    def valid(self, target_classes):
        return (set(self.get_class_count().keys()).intersection(set(target_classes))) and \
               not (set(self.get_class_count().keys()).difference(set(target_classes)))

    def __str__(self):
        return
