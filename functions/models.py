class Models:
    #importing 
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.decomposition import PCA
    from sklearn import svm


    import time
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

    #helper function 
    def evaluate(model, name,X_train, X_test, y_train, y_test):
    
        output = {'model': name}
        start1 = time.time()
        model.fit(X_train, y_train)
        traintime = time.time() - start1
    
        # training metrics
    
        trainpred = model.predict(X_train)
        output['train_precision'] = precision_score(y_train, trainpred)
        output['train_recall'] = recall_score(y_train, trainpred)
        output['train_accuracy'] = accuracy_score(y_train, trainpred)
        output['train_f1'] = f1_score(y_train, trainpred)
        output['train_time'] = traintime

        # testing metrics

        start2 = time.time()
        pred = model.predict(X_test)
        testtime = time.time() - start2

        output['test_precision'] = precision_score(y_test, pred)
        output['test_recall'] = recall_score(y_test, pred)
        output['test_accuracy'] = accuracy_score(y_test, pred)
        output['test_f1'] = f1_score(y_test, pred)
        output['test_time'] = testtime

        # confusion matrix for test set

        conf = pd.crosstab(y_test, pred)

        return output, conf

    #Create a function that executes many Machine Learning Models
    def run_models():
        print('Running models...')
        
        #Using Logistic Regression Algorithm to the Training Set
        lr = LogisticRegression(random_state = 0, max_iter=50000)
        lr_results = evaluate(lr, 'logistic_regression')

        #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
        knn = KNeighborsClassifier(n_neighbors=1)
        knn_results = evaluate(knn, 'knn')

        #print model accuracy on the training data summary
        result_dicts = [knn_results, lr_results, dt_results, 
                    rf_results, nv_results, ada_results, gb_results]
        results = pd.DataFrame([i[0] for i in result_dicts])

        print('_______________________Modeling results_____________________')
        print('df of model accuracy:', results)
        print('Confusion maticies:_________________________________________')

        print('[0] confusion matrix:', lr_results[1])


        print('[1]confusion matrix:', knn_results[1])
    
  
    