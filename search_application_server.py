from qrisp import QuantumDictionary, auto_uncompute, QuantumFloat, QuantumVariable
from qrisp.grover import tag_state, grovers_alg
from qrisp.interface import BackendClient
from qrisp import DefaultBackend
import numpy as np
import string

from flask import Flask, request, jsonify
# To test: curl -X POST http://127.0.0.1:5000/search_word -H "Content-Type: application/json" -d '{"word": "test", "string": "this is a test string"}'

app = Flask(__name__)

# Exact Grover's algorithm

# Generate the quantum oralce encoding the data base
def create_db_oracle(db, labeling, label_size):
    
    qd = QuantumDictionary(
        return_type = QuantumVariable(label_size))
    for i in range(len(db)):
        qd[i] = labeling(db[i])

    def db_oracle(index_qf):
        return qd[index_qf]
    
    return db_oracle


def create_query_oracle(db_oracle, labeling):

    @auto_uncompute
    def query_oracle(index_qf, query_object, phase=np.pi):
        label_bitstring = labeling(query_object)
        label_qv = db_oracle(index_qf)
        tag_state({label_qv : label_bitstring}, phase = phase)
        return

    return query_oracle


class SearchApplication:
    """
    This class implements interfaces for quantum search (search words in an input text) using Grover's algorithm.
    It provides the functions ``update_text`` and ``search_word``.

    Parameters
    ----------
    n : int
        The size, i.e. number of words, of the input text is 2^n.
    k : int
        The label size. A hash function is utilized to represent each word in the input text by its label, 
        i.e. a bitstring of size k. Depending on the number of words 2^n, the label size k must be chosen large enough to avoid collisions.
    backend : BackendClient, optional
        The backend on which to evaluate the quantum circuit. 

    """
     
    def __init__(self, n, k, backend=None):
        self.n = n # Set db size (2**n)
        self.k = k # Set label size 
        
        if backend is None:
            self.backend = DefaultBackend()
        else:
            self.backend = backend

        def labeling(x):
            # Return clipped bitstring of hash
            return bin(hash(x))[-self.k:]

        self.labeling = labeling

        self.data = None
        self.db_oracle = None

    def update_text(self, text):
        """ 
        Update the input text. 
        The text is transformed into an unstructured data base, i.e. a list of string.
        The quantum oracle encoding the data base is generated.

        Parameters
        ----------
        text : str
            The input text.
        
        """
         # Create a translation table to remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        clean_text = text.translate(translator)
        self.data = clean_text.split(" ")[:2**self.n] 

        self.db_oracle = create_db_oracle(self.data, self.labeling, self.k)

    def search_word(self, query_object):
        """
        Search a word in the input text.

        Parameters
        ----------
        query_object : str
            The word searched in the text.

        Returns
        -------
        int
            The position of the query_objct in the text.
        
        """
        
        query_oracle = create_query_oracle(self.db_oracle, self.labeling)

        # Create index integer
        index_qf = QuantumFloat(self.n)
        # Evaluate Grover's algorithm
        grovers_alg(index_qf,
                query_oracle,
                kwargs = {"query_object" : query_object},
                winner_state_amount=1,
                exact=True)
        result = list(index_qf.get_measurement(backend=self.backend,shots=1).keys())[0]
        return result


@app.route('/search_word', methods=['POST'])
def find_position():
    data = request.get_json()

    if not data or 'word' not in data or 'string' not in data:
        return jsonify({
            'error': 'JSON body must include both "word" and "string" keys.'
        }), 400

    word = data['word']
    string = data['string']

    # Use fake backend with a noise model emulating a real quantum computer (based on Qiskit's AerSimulator)
    fake_backend = BackendClient(api_endpoint = "127.0.0.1", port = 42069) 
    
    # Initialize the search application (for n=3, the first 2^3=8 words of the text are relevant)
    app = SearchApplication(3,6,backend=fake_backend) 
    
    # If no backend is specified, the Qrisp DefaultBackend, i.e. the Qrisp simulator is utilized
    #app = SearchApplication(3,6) 
    
    # Generate some sample text
    text = string #"Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur." 
    
    app.update_text(text)
    position = app.search_word(word)

    #position = string.find(word)

    if position == -1:
        return jsonify({
            'message': f"'{word}' not found in the given string.",
            'position': -1
        })

    return jsonify({
        'word': word,
        'string': string,
        'position': position
    })

if __name__ == '__main__':
    app.run(debug=True)
