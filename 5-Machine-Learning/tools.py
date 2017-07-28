#!/usr/bin/python

from sklearn.feature_selection import SelectKBest
def kbest_scores(features, labels, k, list_features):
    kbest = SelectKBest(k = k)
    kbest.fit_transform(features, labels)
    scores = kbest.scores_
    sorted_scores = sorted(zip(list_features[1:], scores), key=lambda x: x[1], reverse=True)
    return sorted_scores


from sklearn.metrics import make_scorer, precision_score, recall_score
def custom_score(features_test, pred):
    p_score = precision_score(features_test, pred)
    r_score = recall_score(features_test, pred)
    if p_score > 0.3 and r_score > 0.3:
        return 1
    return 0


def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    frction = 0.
    if poi_messages == "NaN" or all_messages == "NaN":
        return 0
    else:
        fraction = float(poi_messages)/float(all_messages)
    return fraction


def outlier_cleaner(data_dict, outliers):
    # outliers = {"TOTAL", 
    #            "THE TRAVEL AGENCY IN THE PARK", 
    #            "LOCKHART EUGENE E",
    #            "LAVORATO JOHN J",
    #            "ALLEN PHILLIP K",
    #            "KITCHEN LOUISE",
    #            "WHALLEY LAWRENCE G",
    #            "MCMAHON JEFFREY",
    #            "FALLON JAMES B",
    #            "SHANKMAN JEFFREY A",
    #            "HICKERSON GARY J",
    #            "SHERRIFF JOHN R",
    #            "BAXTER JOHN C",
    # }
    for outlier in outliers:
       data_dict.pop(outlier, 0)

    return data_dict


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import cross_val_score
from time import time

def clf_scores(classifier_list, features_train, labels_train, features_test, labels_test, scoring):
    """Test a list of supervised learning classifiers and return a 
    list of dictionary.
        
        Arg:
            classifier_list: a list of classifiers
        
        Returns: 
            A list of dictionary. Each dictionary contains a mapping
            of keys and scores for the classifier. For example:
            
            {'Classifier': 'KNN',
             'f1': '0.000',
             'mean_accuracy': '0.885',
             'mean_std': '0.027',
             'precision': '0.000',
             'recall': '0.000',
             'time': 0.003}
    """
    results = []
    for name, clf in classifier_list:
        t0 = time()
        clf.fit(features_train, labels_train)
        t1 = round(time()-t0, 3)
        score = cross_val_score(clf, features_test, labels_test, scoring=scoring)
        mean = format(score.mean(), ".3f")
        std = format(score.std(), ".3f" ) 
        labels_pred = clf.predict(features_test)
        pscore = format((precision_score(labels_test, labels_pred)), ".3f")
        rscore = format((recall_score(labels_test, labels_pred)), ".3f")
        fscore = format((f1_score(labels_test, labels_pred)), ".3f")
        # report = classification_report(labels_test, labels_pred)
        results.append({"Classifier": name,
                        "mean_accuracy": mean,
                        "mean_std": std,
                        "time": t1,
                        "precision": pscore,
                        "recall": recall,
                        "f1": fscore
                        # "report": report
                       })
    return results



def clf_report(classifier_list, features_train, labels_train, features_test, labels_test, scoring):
    """Test a list of supervised learning classifiers and return a 
    list of dictionary.
        
        Arg:
            classifier_list: a list of classifiers
        
        Returns: 
            A list of dictionary. Each dictionary contains a mapping
            of keys and scores for the classifier. For example:
            
            {'Classifier': 'KNN',
             'f1': '0.000',
             'mean_accuracy': '0.885',
             'mean_std': '0.027',
             'precision': '0.000',
             'recall': '0.000',
             'time': 0.003}
    """

    for name, clf in classifier_list:        
        t0 = time()
        clf.fit(features_train, labels_train)
        t1 = round(time()-t0, 3)
        score = cross_val_score(clf, features_test, labels_test, scoring=scoring)
        # mean = format(score.mean(), ".3f")
        # std = format(score.std(), ".3f" ) 
        labels_pred = clf.predict(features_test)
        fscore = format((f1_score(labels_test, labels_pred)), ".3f")
        report = classification_report(labels_test, labels_pred)
        print "Time:", t1
        print "Classifier:", name 
        # print "Accuracy mean:", mean
        # print "Accuracy std:", std
        print "F1 score:\n", fscore
        print report

    return None
