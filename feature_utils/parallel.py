from joblib import Parallel, delayed, parallel_backend
from helpers import grouper

class BufferedSOAPFeatures:
    def __init__(self, structures, calculator_params, calculator=SOAP):
        self.X = None
        self.structures = structures
        self.calculator = calculator
        self.calculator_params = calculator_params

    def get_features(self, update_params):
        
        updated_params = self.calculator_params.copy()
        
        for key, value in update_params.items():
            
            if isinstance(value, np.integer):
                value = int(value)
            if isinstance(value, np.floating):
                value = float(value)
            if isinstance(value, np.ndarray):
                value = value.tolist()
                
            updated_params[key] = value

        
        if self.X is None:
            
            #print("Initial calculation")
            self.X = get_features_in_parallel(self.structures,self.calculator,updated_params)
        
        else:
            
            if updated_params == self.calculator_params:
                #print("Stored")
                pass
            else:
                #print("Recalculate")
                self.X = get_features_in_parallel(\
                         self.structures,self.calculator,updated_params)
        
        self.calculator_params = updated_params
        
        return self.X
    

def get_features(frames,calculator,hypers):
    calculatorinstance = calculator(**hypers)
    #print("worker spawned")
    return calculatorinstance.transform(frames).get_features(calculatorinstance)

def get_features_in_parallel(frames,calculator,hypers,blocks=4):
    """helper function that returns the features of a calculator (from calculator.transform())
       in parallel
    """
    
    with parallel_backend(backend="threading"):
        results = Parallel(n_jobs=joblib.cpu_count())(delayed(get_features)(frame, calculator, hypers) for frame in grouper(25,frames))
    
    return np.concatenate(results)