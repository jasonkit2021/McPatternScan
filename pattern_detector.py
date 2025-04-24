# Written by Jason K. in April 2025
# This program uses "deep learning" to scan if an exam (MC questions) has a certain pattern
# Algorithm: it looks back few previous anwsers to guess the next anwser
# Remark 1: the order of answers does matter; the scan does not run reversely without code change
# Remark 2: only A,B,C,D without code change; use "x" to separate different set of data
# Remark 3: cannot detect if the first question has specific value

from tensorflow.keras import layers, models
import numpy as np

CHOICES = ["A","B","C","D"]
LOOK_BACK_COUNT = 8 # look back 8 previous answers to decide the result
LOOKUP = {"A": 0, "B": 1, "C": 2, "D": 3}

# Training data 1 (The teacher doesnt like cat but she will remember "CAT" after thinking her "DAD")
TEST = "DDDDDDAD" # should be "C" after "DAD"
#TEST = "DDDDDDDD" # should have no C
MC_ANSWERS = "BBDABADADCBDAAADBDBADAABADABADABCDABDBBABBAADABDBDAADBBABBABADABBDDABDBBDAADAABDABBAABBDADCADBAAAADABDDBDDADCDDAABDBBADDAADABBAABBADDDABDBDABDAABBCDDBDAABADADCDBAADABAAABDADCBABDDBDBABDADCDADCDCDAABBDDBBAAABDDBBADABDDDDBABDDDDABDADCDABDDDABDDBADDBAABBABADBBABDAAABBDADCDBADAABBABBDAADDABABBDBDAABBDDBAAABBABBDDDADCDDDBDDDAA"

# Data 2
#TEST = "ABCDABCD" # should be C
#TEST = "ABDCABCD" # should be A
#TEST = "AAAAAABC" # should be D (partially match "ABC => D")
#TEST = "xxxxxxxx" # should have no certain ans
#TEST = "xxxxxxxA" # should be B
#MC_ANSWERS = "ABCDABCDCxABDCABCDA"

# Data 3 (JLPT)
#TEST = "ABDABDBA" # long times no C, it should be C!!! [ 8.  9. 77.  6.]
#TEST = "CACACACA" # D [ 0. 28. 10. 62.]
# 2024/12 N4, 2024/12 N2, 2023/07 N2 (Without listening)
#MC_ANSWERS = "ACADCBBDBDACDBCAACBDDBACACDBACABCADCBDBADBCADDBAADCBACABDxDCBABCDBBBCBABDDABCCACDABCBDCACBADDBBDADACBDDBCCBDABBADAAADBDCDABCDBCDCxBDDACACDCDACBBCADDBABDABCDBBACCADBDACDCBCBADBCCCBDADBABDACDCACBDDCDBABC"

# Data 4 (reference against JLPT's pattern) (if some choice happens before, it would more likely to happen again)
#TEST = "BCABBDAB" # B! / B happens forth times, it should be more likely to happen again!
#TEST = "CBABCBAB" # C! / full match case
#MC_ANSWERS = "BCBABCBDBCBABCBBBCBCBABCBABCBABCBCBABCB"

# guess anwsers for the whole paper
# PAPER_FIRST_ANS = "x"

def buildModel(trainInputs, trainLabels):
    model = models.Sequential([
        layers.Flatten(input_shape=(LOOK_BACK_COUNT, len(CHOICES))),  # Flatten 8 previous MC answers, 4 possible choices (A, B, C or D)
        layers.Dense(512, activation='relu'),  # Hidden layer with 512 neurons and ReLU activation
        layers.Dense(64, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation (all options: https://www.geeksforgeeks.org/activation-function-in-tensorflow/)
        layers.Dense(len(CHOICES), activation='softmax') # Output layer with 4 neurons for the 4 classes (A, B, C or D)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(trainInputs, trainLabels, epochs=20)
    return model

def convertLetterToBinary(letter):
    result = []
    for i in range(0, len(CHOICES)):
        result.append(1 if letter == CHOICES[i] else 0)
    return result

def convertWordToBinary(word):
    result = []
    for w in word:
        result.append(convertLetterToBinary(w))
    return np.array([result])

def convertToNnData(trainingDataStr):
    ansHistoryWindowBin = [[0] * len(CHOICES)] * LOOK_BACK_COUNT
    ansHistoryWindowStr = "x" * LOOK_BACK_COUNT
    descs = []
    inputs = []
    labels = []

    for idx, ans in enumerate(trainingDataStr):
        if idx >= len(trainingDataStr):
            break
        if ans == "x": # reached the end of the paper
            ansHistoryWindowBin = [[0] * len(CHOICES)] * LOOK_BACK_COUNT
            ansHistoryWindowStr = "x" * LOOK_BACK_COUNT
            continue

        inputs.append(ansHistoryWindowBin.copy())
        labels.append(LOOKUP[ans])
        descs.append(ansHistoryWindowStr)

        del ansHistoryWindowBin[0]
        ansHistoryWindowBin.append(convertLetterToBinary(ans)[:])
        ansHistoryWindowStr = ansHistoryWindowStr[1:] + ans
    
    inputs = np.array(inputs)
    labels = np.array(labels)
    return (inputs, labels, descs)

'''
def scanIconicData(model, trainInputs, trainDataDescs):
    resultMap = []
    predictions = model.predict_on_batch(trainInputs)

    for idx, prediction in enumerate(predictions):
        resultMap.append((trainDataDescs[idx], np.round(np.max(prediction) * 100, decimals=2), CHOICES[np.argmax(prediction)]))

    resultMap = sorted(
        resultMap, 
        key=lambda x: x[1],
    )

    print('Sure:', resultMap[-5:])
    print('Not sure:', resultMap[:5])

def guessAll(model, firstTry = "", times = 100):
    ansHistoryWindowStr = "x" * LOOK_BACK_COUNT
    result = firstTry
    for _ in range(0, times):
        predictions = model.predict(convertWordToBinary(ansHistoryWindowStr))
        ansHistoryWindowStr = ansHistoryWindowStr[1:] + CHOICES[np.argmax(predictions)]
        result += CHOICES[np.argmax(predictions)]
    print("Guess all:", result)
'''

trainInputs, trainLabels, trainDataDescs = convertToNnData(MC_ANSWERS)

# Build and train the Neural Network
model = buildModel(trainInputs, trainLabels)

# Evaluate the model by the same data
test_loss, test_acc = model.evaluate(trainInputs, trainLabels, verbose=2)
print('Test accuracy:', test_acc)

# Guess
predictions = model.predict(convertWordToBinary(TEST))
print('Guess:', CHOICES[np.argmax(predictions)], 'Probability:', np.round(predictions[0] * 100, decimals=2))

# Find the iconic data
#scanIconicData(model, trainInputs, trainDataDescs)

# Try a paper
#guessAll(model, PAPER_FIRST_ANS)
