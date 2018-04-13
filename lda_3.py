from collections import defaultdict
from random import random, randint
from random import sample as rsample
from random import seed as rseed
from glob import glob
from math import log
import argparse

from nltk.corpus import stopwords
from nltk.probability import FreqDist

from nltk.tokenize import TreebankWordTokenizer
kTOKENIZER = TreebankWordTokenizer()
kDOC_NORMALIZER = True

import time

def dict_sample(d, cutoff=-1):
    """
    Sample a key from a dictionary using the values as probabilities (unnormalized)
    """
    if cutoff==-1:
        cutoff = random()
    normalizer = float(sum(d.values()))
    #print "Normalizer: ", normalizer

    current = 0
    for i in d:
        assert(d[i] > 0)
        current += float(d[i]) / normalizer
        if current >= cutoff:
            #print "Chose", i
            return i
    print("Didn't choose anything: ", cutoff, current)


def lgammln(xx):
    """
    Returns the gamma function of xx.
    Gamma(z) = Integral(0,infinity) of t^(z-1)exp(-t) dt.
    Usage: lgammln(xx)
    Copied from stats.py by strang@nmr.mgh.harvard.edu
    """

    assert xx > 0, "Arg to gamma function must be > 0; got %f" % xx
    coeff = [76.18009173, -86.50532033, 24.01409822, -1.231739516,
             0.120858003e-2, -0.536382e-5]
    x = xx - 1.0
    tmp = x + 5.5
    tmp = tmp - (x + 0.5) * log(tmp)
    ser = 1.0
    for j in range(len(coeff)):
        x = x + 1
        ser = ser + coeff[j] / x
    return -tmp + log(2.50662827465 * ser)


class RandomWrapper:
    """
    Class to wrap a random number generator to facilitate deterministic testing.
    """

    def __init__(self, buff):
        self._buffer = buff
        self._buffer.reverse()

    def __call__(self):
        val = self._buffer.pop()
        print(("Using random value %0.2f" % val))
        return val

class VocabBuilder:
    """
    Creates a vocabulary after scanning a corpus.
    """

    def __init__(self, lang="english", min_length=3, cut_first=100):
        """
        Set the minimum length of words and which stopword list (by language) to
        use.
        """
        self._counts = FreqDist()
        self._stop = set(stopwords.words(lang))
        self._min_length = min_length
        self._cut_first = cut_first

        print(("Using stopwords: %s ... " % " ".join(list(self._stop)[:10])))

##if any word has a len less than 3 or is one of the stop words, ignore it
## runs when initializing the entire program. upon calling a vocab builder object, going to initialize it with these variables, so ever instance has these properties. 


    def scan(self, words):
        """
        Add a list of words as observed.
        """
##building the list. for each word in list of words apssed in, for each word, we're adding that word (where all letters lower), only if not a stopword and len > min len

        for ii in [x.lower() for x in words if x.lower() not in self._stop \
                 and len(x) >= self._min_length]:
            self._counts[ii] += 1


    def vocab(self, size=5000):
        """
        Return a list of the top words sorted by frequency.
        """
        if len(self._counts) > self._cut_first + size:
            #return list(self._counts.keys())[self._cut_first:(size + self._cut_first)]
            rseed(1)
            return [v[0] for v in rsample(sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0])), size)]
        else:
            #return list(self._counts.keys())[:size]
            rseed(1)
            return [v[0] for v in rsample(sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0])), min(len(self._counts), size))]


## self._counts is the number of unique words total. going to look like a dictionary. # of unique words.
# if we have seen more unique words than given size+ cut, do this, otherwise
#adjusting the size of vectors to be standardized. keys should be pre-sorted by frequency. the most freq word (the key) first

## the top one is getting only the keys from the first cut to the size + first cut (such as 100:5100, given size = 5000, cut = 100)

##in the second if we've only hit 500 words, we're gointg to take those 500 words, and give ourselves room up to 500, so that our vocab size is always 5000. always returning a vector of size = size.) 

class LdaTopicCounts:
    """
    This class works for normal LDA.  There is no correlation between words,
    although words can have an aysymmetric prior.
    """

    def __init__(self, beta=0.01):
        """
        Create a topic count with the provided Dirichlet parameter
        """

    ##every time we make this class, going to have a beta dictionary, beta_sum = 0 to start, etc. 
    # a defaultdict is... in a normal dictionary, if you don't explicitly create an element, it's not going to exist. will create a key with whatever class you pass into it. 
    #every time you call a new element we're going to create a freq distribution if you dn't assign it immediately

        self._beta = {}
        self._beta_sum = 0.0

        # Maintain a count for each word
        self._normalizer = FreqDist()
        self._topic_term = defaultdict(FreqDist)
        self._default_beta = beta

        self._finalized = False

    def set_vocabulary(self, words):
        """
        Sets the vocabulary for the topic model.  Only these words will be
        recognized.
        """
        for ii in range(len(words)):
            self._beta_sum += self._default_beta

        ### unclear why we're doing this!!! (adding a dictionary into an integer. where is default_beta being changed?)

    def change_prior(self, word, beta):
        """
        Change the prior for a single word.
        """

        ##prior is the prior probability. given no info about the context of the word, what's the p that we're going to have it with this meaning or this use case
        
        assert not self._finalized, "Priors are fixed once sampling starts."

        self._beta[word] = beta
        self._beta_sum += (beta - self._default_beta)

    def initialize(self, word, topic):
        """
        During initialization, say that a word token with id ww was given topic
        """
        ## every time you see a word in this topic, put it in the dictionary (using the defaultdictionary property), finding the freq distribution for each one we pass in, and if it's the first itme we've seen it, creating a freq distrbitution. the normalizer is how many times we're seeing this topic 

        self._topic_term[topic][word] += 1
        self._normalizer[topic] += 1

    def change_count(self, topic, word, delta):
        """
        Change the topic count associated with a word in the topic
        """
        
        self._finalized = True

        self._topic_term[topic][word] += delta
        self._normalizer[topic] += delta

    def get_normalizer(self, topic):
        """
        Return the normalizer of this topic
        """
        return self._beta_sum + self._normalizer[topic]

    def get_prior(self, word):
        """
        Return the prior probability of a word.  For tree-structured priors,
        return the probability marginalized over all paths.
        """
        return self._beta.get(word, self._default_beta)
    
    def get_observations(self, topic, word):
        """
        Return the number of occurences of a combination of topic, word, and
        path.
        """
        return self._topic_term[topic][word]

    def word_in_topic(self, topic, word):
        """
        Return the probability of a word type in a topic
        """

## for one topic, all the vals for each word will add to one. therefore were getting the p of that word in the topic, against all other words. could have integers to represent this, but not as useful. 

        val = self.get_observations(topic, word) + self.get_prior(word)
        val /= self.get_normalizer(topic)
        return val
    

    def report(self, vocab, handle, limit=25):
        """
        Create a human readable report of topic probabilities to a file.
        """

    ## kk is an undefined iterator. is an element of normalizer. 
    ## ww is a word token
    ## self._normalizer is the freqdist()

# for each pair in the freqdist normalizer. kk is a (word, frequency) pair

        for kk in self._normalizer:
            normalizer = self.get_normalizer(kk)
            handle.write("------------\nTopic %i (%i tokens)\n------------\n" % \
                      (kk, self._normalizer[kk]))

            word = 0
            topic_terms = self._topic_term[kk]
            sorted_tt = sorted(topic_terms, key=lambda x: topic_terms[x], reverse=True)
            #for ww in self._topic_term[kk]:
            for ww in sorted_tt:
                handle.write("%0.5f\t%0.5f\t%0.5f\t%s\n" % \
                             (self.word_in_topic(kk, ww),
                              self.get_observations(kk, ww),
                              self.get_prior(ww),
                              vocab[ww].encode("UTF-8")))
                      
                word += 1
                if word > limit:
                    break


class Sampler:
    def __init__(self, num_topics, vocab, alpha=0.1, beta=0.01, rand_stub=None):
        """
        Create a new LDA sampler with the provided characteristics
        """
        self._num_topics = num_topics
        self._doc_counts = defaultdict(FreqDist)
        self._doc_tokens = defaultdict(list)
        self._doc_assign = defaultdict(list)
        self._alpha = [alpha for x in range(num_topics)]
        self._sample_stats = defaultdict(int)
        self._vocab = vocab
        self._topics = LdaTopicCounts(beta)
        self._topics.set_vocabulary(vocab)
        self._lhood = []
        self._time = []
        self._rand_stub = rand_stub

    def change_alpha(self, idx, val):
        """
        Update the alpha value; note that this invalidates precomputed values.
        """
        self._alpha[idx] = val


    def get_doc(self, doc_id):
        """
        Get the data associated with an individual document
        """
        return self._doc_tokens[doc_id], self._doc_assign[doc_id], \
            self._doc_counts[doc_id]

    ##returning the freq distribution for that document:
            ## (list of tokens for the doc, id for the doc, freq dist for that doc)


    def add_doc(self, doc, vocab, doc_id = None, 
                token_limit=-1):
        """
        Add a document to the corpus.  If a doc_id is not supplied, a new one
        will be provided.
        """

    ##index is a returning the index of that token in a list. vocab is the lst we made earlier, find the location of x. 
        #for word in a doc, if we're familiar with the word, save its location. if not continue. transforming doc from words into numbers. in doc we are tracking the word as a number represenatation. will return the index of that word.


        temp_doc = [vocab.index(x) for x in doc if x in vocab]
        #print '\n\n', temp_doc, '\n\n'

        if not doc_id:
            doc_id = len(self._doc_tokens)
        assert not doc_id in self._doc_tokens, "Doc " + str(doc_id) + \
            " already added"

        if len(temp_doc) == 0:
            print("WARNING: empty document (perhaps the vocab doesn't make sense?)")
        else:
            self._doc_tokens[doc_id] = temp_doc

        token_count = 0
        for ww in temp_doc:
            assignment = randint(0, self._num_topics - 1)
            self._doc_assign[doc_id].append(assignment)
            self._doc_counts[doc_id][assignment] += 1
            self._topics.initialize(ww, assignment)

            token_count += 1
            if token_limit > 0 and token_count > token_limit:
                break 
                #if we've hit the number of tokens we want, break, stop adding tokens/topics

        assert len(self._doc_assign[doc_id]) == len(temp_doc), \
               "%s != %s" % (str(self._doc_assign[doc_id]), str(temp_doc))

        ## assert -- makes sure we're on track. if the len of the doc here matches the len of the temp doc and the two strings do not match, then we're good. otherwise, we've made a mistake in computation or planinng and the assumptions we've held are not accurate. if false, i fucked up. we never want assert to fail!
                                                          
        return doc_id


    def change_topic(self, doc, index, new_topic):
        """
        Change the topic of a token in a document.  Update the counts
        appropriately.  -1 is used to denote "unassigning" the word from a topic.
        """
        assert doc in self._doc_tokens, "Could not find document %i" % doc
        assert index < len(self._doc_tokens[doc]), \
            "Index %i out of range for doc %i (max: %i)" % \
            (index, doc, len(self._doc_tokens[doc]))

        term = self._doc_tokens[doc][index]  ## term still specific word. the actual token/word, not the number representation
        alpha = self._alpha    ## alpha still the list

        assert index < len(self._doc_assign[doc]), \
               "Bad index %i for document %i, term %i %s" % \
               (index, doc, term, str(self._doc_assign[doc]))

        old_topic = self._doc_assign[doc][index]  # might have to set new_topic.


        #terms to topics, documents to topics, specific assignments
            
        if old_topic != -1:
             ## -1 means its unassigned, new topic is passed in, hasn't been motified yet.  if we have an old topic, want to change it to -1 to reduce the counts of "we now have one less term in that topic, one less value indicating a document value for that topic. if new topic exists, will do that second change, making a new assignment, increasing the count."

            assert new_topic == -1
            
            # TODO: Add code here to keep track of the counts and
            # assignments
            self._doc_assign[doc][index] = -1
            ## unassign input values for doc and index above. this is a list, w/in doc this is the index of the word we are looking for and then we decrement it. 

            self._doc_counts[doc][old_topic] -= 1
            ##assignment changes to old_topic here

            self._topics.change_count(old_topic, term, delta = -1)            
            ## changing the freq distribution of the old topic for that term and reducing it by 1


        if new_topic != -1:
            assert old_topic == -1

            # TODO: Add code here to keep track of the counts and
            # assignments
            self._doc_assign[doc][index] = new_topic 
            self._doc_counts[doc][new_topic] += 1
            self._topics.change_count(new_topic, term, delta = 1)


    def run_sampler(self, iterations = 100):
        """
        Sample the topic assignments of all tokens in all documents for the
        specified number of iterations.
        """
        for ii in range(iterations):
            #if ii % 20 == 0:
            #    print("Iteration %i" % ii)
            start = time.time()
            for jj in self._doc_assign:
                self.sample_doc(jj)

            total = time.time() - start
            lhood = self.lhood()
            print(("Iteration %i, likelihood %f, %0.5f seconds" % (ii, lhood, total)))
            ##given our float, we only want the first five indexes. the pink % is the linker. "ive finished my string, link it to the vars [the %'s --integers] in order here". %s = string

            self._lhood.append(lhood)
            self._time.append(total)

    def report_topics(self, vocab, outputfilename, limit=10):
        """
        Produce a report to a file of the most probable words in a topic, a
        history of the sampler, and the state of the Markov chain.
        """
        topicsfile = open(outputfilename + ".topics", 'w')
        self._topics.report(vocab, topicsfile, limit)

        statsfile = open(outputfilename + ".stats", 'w')
        tmp = "iter\tlikelihood\ttime(s)\n"
        statsfile.write(tmp)
        for it in range(0, len(self._lhood)):
            tmp = str(it) + "\t" + str(self._lhood[it]) + "\t" + str(self._time[it]) + "\n"
            statsfile.write(tmp)
        statsfile.close()

        topicassignfile = open(outputfilename + ".topic_assign", 'w')
        for doc_id in list(self._doc_assign.keys()):
            # print self._doc_assign[doc_id]
            tmp = " ".join([str(x) for x in self._doc_assign[doc_id]]) + "\n"
            topicassignfile.write(tmp)
        topicassignfile.close()

        doctopicsfile = open(outputfilename + ".doc_topics", 'w')
        for doc_id in list(self._doc_counts.keys()):
            tmp = ""
            for tt in range(0, self._num_topics):
                tmp += str(self._doc_counts[doc_id][tt]) + " "
            tmp = tmp.strip()
            tmp += "\n"
            doctopicsfile.write(tmp)
        doctopicsfile.close()

    def sample_probs(self, doc_id, index):
        """
        Create a dictionary storing the conditional probability of this token being assigned to each topic.
        """
        assert self._doc_assign[doc_id][index] == -1, \
          "Sampling doesn't make sense if this hasn't been unassigned."
        
##making sure this has been unassigned, create an empty dict, collect the doc tokens for this doc at this index (get one token)

        sample_probs = {}
        term = self._doc_tokens[doc_id][index]

        ## TODO: Compute the conditional probability of
        ## sampling a topic; at the moment it's just the
        # # uniform probability.

        #     n_dk_left = self._doc_counts[doc_id][kk]
        #     alpha_k_left = self._alpha[kk]
        #     Sigma_n_di_a_i = sum(self._doc_counts[doc_id].values()) + sum(self._alpha)
        #     right_word_in_topic = self._topics.word_in_topic(kk, term)

        #     sample_probs[kk] = (float(n_dk_left + alpha_k_left) / float(Sigma_n_di_a_i)) * right_word_in_topic

        # return sample_probs

        for kk in range(self._num_topics):
            # TODO: Compute the conditional probability of
            # sampling a topic; at the moment it's just the
            # uniform probability.
            sample_probs[kk] = 1.0 / float(self._num_topics)

            # # n_dk_left = self._doc_counts[doc_id][kk]
            # # alpha_k_left = self._alpha[kk]
            # Sigma_n_di_a_i = sum(self._doc_counts[doc_id].values()) + sum(self._alpha)
            # right_word_in_topic = self._topics.word_in_topic(kk, term)

            sample_probs[kk] = (float(self._doc_counts[doc_id][kk] + self._alpha[kk]) / float(sum(self._doc_counts[doc_id].values()) + sum(self._alpha))) * self._topics.word_in_topic(kk, term)

        return sample_probs
        

    def sample_doc(self, doc_id, debug=False):
        """
        For a single document, compute the conditional probabilities and
        resample topic assignments.
        """

        one_doc_topics = self._doc_assign[doc_id] 
        ##getting topic dist for doc doc_id, which correlates to Pi
        
        topics = self._topics
        ##topics is going to have the word dist, the normalizer against how much we're seeing that topic. 

        #for each topic in the document, change the topic of word index in  document_id to - 1. throwing out our guess from before

        for index in range(len(one_doc_topics)):
            self.change_topic(doc_id, index, -1)
            sample_probs = self.sample_probs(doc_id, index)

            ## getting the list of probabilties that a word is of this topic in this document 

            if self._rand_stub:
                cutoff = self._rand_stub()  ## a randomly generated number. 
            else:
                cutoff = random()  #either chosiing in advanced when to cut off or we're doing it now
            new_topic = dict_sample(sample_probs, cutoff)
            ##new topic prediction. the token is the key, the value is a probability

            self.change_topic(doc_id, index, new_topic)

        return self._doc_assign[doc_id]


    def lhood(self):
        val = self.doc_lhood() + self.topic_lhood()
        return val

    def doc_lhood(self):  ## doc_likelihood is going to correspond to Pi
        doc_num = len(self._doc_counts)
        alpha_sum = sum(self._alpha)

        val = 0.0
        val += lgammln(alpha_sum) * doc_num
        tmp = 0.0
        for tt in range(0, self._num_topics):
            tmp += lgammln(self._alpha[tt])
        val -= tmp * doc_num
        for doc_id in self._doc_counts:
            for tt in range(0, self._num_topics):
                val += lgammln(self._alpha[tt] + self._doc_counts[doc_id][tt])
            val -= lgammln(alpha_sum + len(self._doc_assign[doc_id]))

        return val

    def topic_lhood(self):  ## topic likelihood corresponds to B
        val = 0.0
        vocab_size = len(self._vocab)

        val += lgammln(self._topics._beta_sum) * self._num_topics
        val -= lgammln(self._topics._default_beta) * vocab_size * self._num_topics
        for tt in range(0, self._num_topics):
            for ww in self._vocab:
                val += lgammln(self._topics._default_beta + self._topics._topic_term[tt][ww])
            val -= lgammln(self._topics.get_normalizer(tt))
        return val
    


def tokenize_file(filename):
    contents = open(filename, encoding="utf-8").read()
    for ii in kTOKENIZER.tokenize(contents):
        yield ii
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--doc_dir", help="Where we read the source documents",
                           type=str, default=".", required=False)
    argparser.add_argument("--language", help="The language we use",
                           type=str, default="english", required=False)
    argparser.add_argument("--output", help="Where we write results",
                           type=str, default="result", required=False)    
    argparser.add_argument("--vocab_size", help="Size of vocabulary",
                           type=int, default=3000, required=False)
    argparser.add_argument("--num_topics", help="Number of topics",
                           type=int, default=10, required=False)
    argparser.add_argument("--num_iterations", help="Number of iterations",
                           type=int, default=1000, required=False)    
    args = argparser.parse_args()

    vocab_scanner = VocabBuilder(args.language)

    # Create a list of the files
    search_path = "%s/*.txt" % args.doc_dir
    files = glob(search_path)
    assert len(files) > 0, "Did not find any input files in %s" % search_path
    
    # Create the vocabulary
    for ii in files:
        vocab_scanner.scan(tokenize_file(ii))

    # Initialize the documents
    vocab = vocab_scanner.vocab(args.vocab_size)
    print((len(vocab), vocab[:10])) 
    lda = Sampler(args.num_topics, vocab)
    for ii in files:
        lda.add_doc(tokenize_file(ii), vocab)

    lda.run_sampler(args.num_iterations)
    lda.report_topics(vocab, args.output, 20)

