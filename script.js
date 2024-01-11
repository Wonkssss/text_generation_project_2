const TextGenBtn = document.getElementById('TextGen');
let model;

stoi = {'\t': 0, '\n': 1, ' ': 2, '!': 3, '"': 4, '#': 5, '$': 6, '%': 7, '&': 8, "'": 9, '(': 10, ')': 11, '*': 12, '+': 13, ',': 14, '-': 15, '.': 16, '/': 17, '0': 18, '1': 19, '2': 20, '3': 21, '4': 22, '5': 23, '6': 24, '7': 25, '8': 26, '9': 27, ':': 28, ';': 29, '=': 30, '?': 31, 'A': 32, 'B': 33, 'C': 34, 'D': 35, 'E': 36, 'F': 37, 'G': 38, 'H': 39, 'I': 40, 'J': 41, 'K': 42, 'L': 43, 'M': 44, 'N': 45, 'O': 46, 'P': 47, 'Q': 48, 'R': 49, 'S': 50, 'T': 51, 'U': 52, 'V': 53, 'W': 54, 'X': 55, 'Y': 56, 'Z': 57, '[': 58, '\\': 59, ']': 60, '_': 61, '`': 62, 'a': 63, 'b': 64, 'c': 65, 'd': 66, 'e': 67, 'f': 68, 'g': 69, 'h': 70, 'i': 71, 'j': 72, 'k': 73, 'l': 74, 'm': 75, 'n': 76, 'o': 77, 'p': 78, 'q': 79, 'r': 80, 's': 81, 't': 82, 'u': 83, 'v': 84, 'w': 85, 'x': 86, 'y': 87, 'z': 88, '{': 89, '|': 90, '}': 91, '~': 92, '\xa0': 93, '©': 94, '®': 95, '°': 96, '´': 97, '·': 98, 'Á': 99, 'Ä': 100, 'Ç': 101, 'É': 102, 'Ñ': 103, 'Ô': 104, 'Ö': 105, 'ß': 106, 'á': 107, 'ä': 108, 'ç': 109, 'è': 110, 'é': 111, 'ë': 112, 'í': 113, 'î': 114, 'ï': 115, 'ñ': 116, 'ó': 117, 'ö': 118, 'ø': 119, 'ú': 120, 'û': 121, 'ü': 122, 'Ā': 123, 'ā': 124, 'Ď': 125, 'ď': 126, 'ğ': 127, 'Ġ': 128, 'Ō': 129, 'ō': 130, 'ş': 131, 'Ţ': 132, 'ū': 133, 'ǣ': 134, 'ș': 135, 'Ț': 136, 'ə': 137, 'ɨ': 138, 'ɪ': 139, 'ʾ': 140, 'ˈ': 141, '˝': 142, 'Θ': 143, 'Σ': 144, 'ε': 145, 'З': 146, 'К': 147, 'Л': 148, 'П': 149, 'С': 150, 'Т': 151, 'а': 152, 'в': 153, 'г': 154, 'д': 155, 'е': 156, 'з': 157, 'л': 158, 'м': 159, 'н': 160, 'о': 161, 'р': 162, 'т': 163, 'ы': 164, 'ь': 165, 'я': 166, 'ё': 167, 'أ': 168, 'ا': 169, 'ب': 170, 'ت': 171, 'ح': 172, 'ر': 173, 'س': 174, 'ع': 175, 'غ': 176, 'ك': 177, 'ل': 178, 'ن': 179, 'و': 180, 'ي': 181, '\u2003': 182, '\u200b': 183, '\u200e': 184, '–': 185, '—': 186, '‘': 187, '’': 188, '‚': 189, '“': 190, '”': 191, '†': 192, '•': 193, '…': 194, '∞': 195, '▪': 196, 'の': 197, 'び': 198, 'ア': 199, 'カ': 200, 'ド': 201, 'ー': 202, '叫': 203, '巻': 204, '心': 205, '拳': 206, '犬': 207, '王': 208, '駄': 209, '︎': 210, '\ufeff': 211, '（': 212, '，': 213, '�': 214}
itos = Object.keys(stoi).reduce((acc, key) => {
    acc[stoi[key]] = key;
    return acc;
}, {});

// Initialize ONNX model
async function initModel() {
    model = await ort.InferenceSession.create('model.onnx');
    console.log('model successfully loaded');
}


// Preprocess input text
function preprocess(inputText) {

    // Split input text into array of characters
    const inputTextSplitted = inputText.split('');
    // Map each character to its corresponding index from the vocabulary
    const inputTextEncoded = inputTextSplitted.map(char => {
        return stoi[char];
    });

    //inputTextEncoded = inputTextEncoded.slice(-1); // get only the last character !!!

    // Return encoded input text
    return inputTextEncoded;

}

function softmax(arr) {
    console.log(arr)
    const expArr = arr.map((value )=> Math.exp(value));
    const sumExp = expArr.reduce((acc, val) => acc + val, 0);
    return expArr.map((value) => value / sumExp);
  }
  
  function multinomialSampling(probs) {
    // Apply softmax to the probabilities
    const softmaxProbs = softmax(probs);
  
    // Generate a random number between 0 and 1
    const randomValue = Math.random();
  
    // Perform multinomial sampling
    let cumulativeProb = 0;
    for (let i = 0; i < softmaxProbs.length; i++) {
      cumulativeProb += softmaxProbs[i];
      if (randomValue <= cumulativeProb) {
        return i; // Return the index of the selected element
      }
    }
    // This should not happen, but just in case
    return softmaxProbs.length - 1;
  }
  

// Postprocess output text
function postprocess(outputText) {

    multin = multinomialSampling(outputText);
    console.log('multin: ', multin);
    multin = itos[multin];
    //multin = multin.slice(0, -1); // get only the last character !!!
    return multin;
}


// Generate text
async function generateText() {
    
    // Get input text
    let inputText = document.getElementById('inputText').value;

    for (let i = 0; i < 500; i++) {

    // Preprocess input text
    let inputTextPreprocessed = preprocess(inputText);
    console.log('inputTextPreprocessed: ', inputTextPreprocessed) //----------------------------------------------

    // Get input text length
    let inputTextLength = inputTextPreprocessed.length;
    console.log('inputTextLength: TEXT LENGTH !!! : ', inputTextLength) //----------------------------------------------

    // Create input tensor
    let inputTensor = new ort.Tensor(
        'int64',
        inputTextPreprocessed,
        [1, inputTextLength]
    );

    console.log('input text: ', inputText);
    // Run inference with ONNX model
    let output = await model.run({ input: inputTensor });
    console.log('output: ', output)

    // Get output tensor
    let outputTensor = output.output;
    console.log('output tensor: !!! ', outputTensor.data);
    // get only the last value of the output tensor
    console.log("-outputTensor.data.length/inputTextLength: ", outputTensor.data.length, inputTextLength)
    last_logits = outputTensor.data.slice(-outputTensor.data.length/inputTextLength); //----------------------------------------------
    console.log('last_logits: ', last_logits);

    // Postprocess output text
    let outputTextPostprocessed = postprocess(last_logits);
    inputText += outputTextPostprocessed;

    console.log('output here: ', outputTextPostprocessed);
    // Display output text
    // document.getElementById('outputText').innerHTML = outputTextPostprocessed;
    document.getElementById('outputText').innerHTML = inputText;
    }

}

initModel().then(() => { // Make sure the model is loaded before running the inference

    //generateText();
    TextGenBtn.addEventListener('click', generateText);
});

// "Hello my name is Jack"
