""" 
A Bayesian and SVM Spam Filter by Nicholas Shelly
Date: 3 Jan 2012
The primary classification uses a Naive Bayesian model, with auto-learning capability 
based on email with really high or really low spamtiscity.
Compares the results to a more reliable but slower Support Vector Machine (SVM) classifier.
""" 
import sys
import random
import re
import tarfile
import operator
import os
try:
    from svmutil import *
    SVM = 1
except ImportError:
    SVM = 0
    print "Could not find svm suite.  Please download here: http://bit.ly/2rwxl"

DATA_DIR='data'
HAM_FILES = ['20030228_hard_ham.tar.bz2',]
SPAM_FILES = ['20030228_spam_2.tar.bz2', '20050311_spam_2.tar.bz2']

if len(sys.argv) > 1:
    DEBUG = sys.argv[1]
else:
    DEBUG = 0

class SpamFilter:
    """ A naive Bayesian and SVM spam filter """
    
    def __init__(self, options=None):
        self.ham = {}
        self.spam = {}
        self.common = {}
        self.svm_features = []
        self.email_data = []

        # Options for naive Bayesian 
        self.cutoff = 0.75      # Cutoff for classifying as spam (higher, fewer false positive)
        self.count_threshold = 20   # Number of times token appears to be relevant
        self.size_threshold = 6     # The token '1c' is less significant then 'einstein'
        self.unique_threshold = 3   # Unique characters in string ('qqqq' => 1)
        self.prob_threshold = 0.1   # Before considering token significant
        self.lower = False          # Whether or not to normalize to all lowercase
        self.prob_spam = 0.45
        self.num_tokens = 15

        # SVM Feature options
        self.svm_prob_threshold = 0.2         # Before considering feature significant
        self.svm_count_threshold = 30         # Number of times string appears

        # Auto-learning options
        self.auto_learn = True
        self.taoplus = 0.98           # Threshold for exemplar spam for auto-learning
        self.taominus = 0.05          # Threshold for exemplar ham for auto-learning

        # Stats
        self.exemplar_spam = 0
        self.exemplar_ham = 0
        self.false_positives = 0
        self.true_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        
    def _error(self, data, classifier):
        """ Returns the error rate of a classifier on training or development data """
        errors = 0
        for X,y in data:
            c = classifier(X)
            if c != y:
                errors += 1
            if c and not y:
                self.false_positives += 1
            elif c and y:
                self.true_positives += 1
            elif not c and y:
                self.false_negatives += 1
            elif not c and not y:
                self.true_negatives += 1

        return errors/float(len(data))

    def _get_tokens(self, content):
        """ Treat dollar signs, apostrophes and dashes as part of words;
            everything else as a token separator """
        tokens = re.findall(r'[\w$-\']+', content)
        tokens = [t.lower() for t in tokens]
        for t in tokens:
            if (len(set(t)) < self.unique_threshold or len(t) < self.size_threshold):
                del t
        return tokens

    def _parse_email(self, table, content):
        """ Parse email, adding tokens to appropriate dictionary """
        tokens = self._get_tokens(content)
        for t in tokens:
            if t not in table:
                table[t] = 0
            table[t] += 1
            if t not in self.common:
                self.common[t] = 0
            self.common[t] += 1
            # Future optimizations:
            # Check for urls, header tags, breadth of words (number of appearances), etc 

    def load_corpus(self, files, spam=True, data_dir=DATA_DIR):
        """ 
        Loads corpus into the spam or ham dictionary, as well as a common dictionary
        """
        for fname in files:
            if os.path.exists(fname):
                # File located in current directory
                path_to_file = fname 
            else:
                path_to_file = os.path.join(data_dir, fname)
            print "Reading tarfile  %-30s ..." % (path_to_file),
            if tarfile.is_tarfile(path_to_file):
                with tarfile.open(path_to_file, 'r|bz2') \
                    as tar:
                    for member in tar:
                        if member.isfile():
                            f=tar.extractfile(member)
                            content=f.read()
                            if spam:
                                self.email_data.append((content, 1))
                            else:
                                self.email_data.append((content, 0))
                            
                    print "done, read %d emails." % len(tar.getmembers())
                                
            else:
                print "Unable to read file '%s/%s'..." % (data_dir, fname)


    def train_data(self, training_percent=33.0):
        N = len(self.email_data)
        training_indices = set(random.sample(range(N), \
                                             int(N*training_percent/100.0)))
        self.training = []
        self.development = []
        for i in range(N):
            if i in training_indices:
                self.training.append(self.email_data[i])
            else:
                self.development.append(self.email_data[i])
        for email in self.training:
            if email[1]:
                # Add words to spam dictionary
                self._parse_email(self.spam, email[0])
            else:
                # Add words to ham dictionary
                self._parse_email(self.ham, email[0])

    def _classify_bayes_test(self, content):
        """
        When auto-learning enabled, re-train when identifying exemplar spam (> taoplus) 
        or ham emails (< taominus)
        """
        if self.auto_learn:
            return self._classify_bayes(content, auto_learn=True)

    def _debug_content(self, content, pspam, plist):
        if DEBUG:
            print "%f: \n%s" % (pspam, "\n".join(map(str,plist[:self.num_tokens])))
            print "Test: P(SPAM|email) = %f" % pspam
            ans = raw_input("View email (y/[N])? ")
            if ans in ('y','Y'):
                print content
                raw_input('ok?')

    def _classify_bayes(self, content, auto_learn=False):
        """
        Calculates probability email is spam based on top N words of content:
        P(S|w1,w2,..wN) = P(w1|S)*P(w2|S)*...P(wN|S) /      \
                        [ P(w1|S)*P(w2|S)*...P(wN|S) + (1-P(w1|S)*(1-P(w2|S)*...(1-P(wN|S) ]
        """
        tokens = self._get_tokens(content)
        plist = []
        for t in set(tokens):
            pword = self.calculate_probability_spam(t)
            if pword:
                plist.append(pword)
        # Sort by most significant words (likely spam or ham)
        plist.sort(key=lambda p: abs(0.5-p[1]), reverse=True)
        numerator = 1.0
        denominator_left = 1.0
        denominator_right = 1.0
        for i in xrange(min(self.num_tokens, len(plist))):
            numerator *= plist[i][1]
            denominator_left *= plist[i][1]
            denominator_right *= 1.0 - plist[i][1]
        pspam = numerator / (denominator_left + denominator_right)

        if auto_learn:
            # Auto-learn ('iterative') if very likely spam or very likely ham
            if pspam > self.taoplus:
                # Add words to spam dictionary
                self._parse_email(self.spam, content)
                self.exemplar_spam += 1
                self._debug_content(content, pspam, plist)

            elif pspam < self.taominus:
                # Add words to ham dictionary
                self._parse_email(self.ham, content)
                self.exemplar_ham += 1
                self._debug_content(content, pspam, plist)

        return 1 if pspam > self.cutoff else 0 
        
    def calculate_probability_spam(self, token):
        """
        Calculate Bayesian probability of spam given token, P(S|w), meeting minimum criteria
        P(S|w) = P(w|S)*P(S) / [ P(w|S)*P(S) + P(w|H)*P(H) ]
        """
        if len(token) >= self.size_threshold and len(set(token)) > self.unique_threshold:
            b = self.spam[token] if token in self.spam else 1 
            g = self.ham[token] if token in self.ham else 1 
            if b+g > self.count_threshold:
                numerator = b*1.0/len(self.spam) * self.prob_spam
                denominator = b*1.0/len(self.spam) * self.prob_spam + \
                              g*1.0/len(self.ham) * (1. - self.prob_spam)

                pspam_word = numerator / denominator
                if abs(0.5 - pspam_word) > self.prob_threshold:
                    return (token, pspam_word, b+g)
        return None
        
    def _set_svm_features(self):
        """ Get SVM features """
        for token in self.common:
            probability_spam = self.calculate_probability_spam(token)
            if  probability_spam \
                and probability_spam[2] >= self.svm_count_threshold \
                and abs(0.5-probability_spam[1]) >= self.svm_prob_threshold:
                # Limit the number of SVM features, for a quicker build
                self.svm_features.append(token)

    def _get_svm_classifier(self, data, options='-c 1 -t 0'):
        """ Returns SVM classifer based on svmlib model """

        labels = [y for (X,y) in data]
        samples = [X for (X,y) in data]

        # Return classification value for testing 
        svm_model.predict = lambda self, x: svm_predict([0], [x], self, '-q')[0][0]
        problem = svm_problem(labels, samples)
        param = svm_parameter(options)
        model = svm_train(problem, param)
        return model.predict

    def _build_svm_features(self, email):
        """ 
        Builds a binary feature vector where each feature is 1 if the corresponding 
        dictionary words is in the review, 0 if not.
        """
        features = []
        email_tokens = self._get_tokens(email)
        for token in self.svm_features:
            if token in email_tokens:
                features.append(1)
            else:
                features.append(0)
        return features

    def print_strong_words(self):
        """ Prints the most significant words, indicating spam or not spam """
        strong_words = []
        for token, value in self.common.items():
            probability_spam = self.calculate_probability_spam(token)
            if probability_spam:
                strong_words.append(probability_spam)
        strong_words.sort(key=operator.itemgetter(1))
        print "\nTop words most likely to be spam:"
        for s in strong_words[:-10:-1]:
            print "%s = %f, %d occurrences" % (s[0], s[1], s[2])
        print "\nTop words least likely to be spam:"
        for s in strong_words[:10]:
            print "%s = %f, %d occurences" % (s[0], s[1], s[2])
          
    def print_stats(self):
        self.num_spam = [i[1] for i in self.email_data].count(1)
        print "Read %d emails," % len(self.email_data), 
        print "{:.2%} spam ".format(self.num_spam * 1.0 / len(self.email_data))
        self.print_strong_words()
        print "\nVariables:"
        print "tao- = %f" % self.taominus
        print "tao+ = %f" % self.taoplus
        print "Prob(Spam) = %f" % self.prob_spam
        print "Spam cutoff = %f" % self.cutoff
        print "Count minimum = %d" % self.count_threshold
        print "\n########################################################"
        print "Naive Bayes:"
        print "%d unique tokens" % len(self.common)
        print "Training error: ", self._error(self.training, self._classify_bayes)
        print "Development error: ", self._error(self.development, \
                                                 self._classify_bayes_test)
        print "Auto-learning on %d spam and %d ham" % \
                (self.exemplar_spam, self.exemplar_ham)
        print "False positives = %.2f%%" % \
                (self.false_positives * 1.0 / (self.false_positives + self.true_positives))
        print "False negatives = %.2f%%" % \
                (self.false_negatives * 1.00 / (self.false_negatives + self.true_negatives))

        self.false_positives = self.true_positives = \
                self.false_negatives = self.true_negatives = 0

        if SVM:
            print "\n########################################################"
            self._set_svm_features()
            print "SVM (%d features):" % len(self.svm_features)
            print "Building training samples..."
            training_samples = [(self._build_svm_features(email[0]), email[1]) \
                                for email in self.training]
            classify_svm = self._get_svm_classifier(training_samples, options='-c 1 -t 0')
            print "Training error:", self._error(training_samples, classify_svm)
            print "Building development samples..."
            development_samples = [(self._build_svm_features(email[0]), email[1]) \
                                for email in self.development]
            print "Development error:", self._error(development_samples, classify_svm)
            print "False positives = %.f%%" % \
                    (self.false_positives * 1.0 / (self.false_positives + self.true_positives))
            print "False negatives = %.1f%%" % \
                    (self.false_negatives * 1.00 / (self.false_negatives + self.true_negatives))
            print "\n" * 5
            print "########################################################"
        
def main():
    sf = SpamFilter()
    sf.load_corpus(HAM_FILES, spam=False)
    sf.load_corpus(SPAM_FILES, spam=True)
    sf.train_data(training_percent=50)
    sf.print_stats()

if __name__=="__main__":
    main()
    sys.exit()
