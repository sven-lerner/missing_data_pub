import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import json
import time
from datetime import datetime


class PaperResult(ABC):
    
    @property
    @abstractmethod
    def section(self) -> str:
        "the section of the paper the result corresponds to"
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        "a name for the result, will be used as the save_path"
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        "a description of the result, will be stored as metatdata along with the time of writing the file"
        pass


    @abstractmethod
    def run(self, loud=True, **kwargs):
        pass
    
    @abstractmethod
    def setup(self, **kwargs):
        '''put all setup for the class in here, but do not call plot or show'''
        pass
    
    
class PaperPlot(PaperResult, ABC):
    save_dir = '../images-pdfs'
    
    def run(self, loud=True):
        save_loc = f"{self.save_dir}/{self.section}/{self.name}"
        print(f"saving to save_loc")
        plt.savefig(f'{save_loc}.pdf', bbox_inches='tight')
        with open(f'{save_loc}.json', 'w') as f:
            json.dump({'date_updated': datetime.now().strftime("%m/%d/%Y,%H:%M:%S"),
                         'name': self.name,
                         'description':self.description}, f)
        if loud:
            plt.show()
        else:
            plt.close()

class PaperTable(PaperResult, ABC):
    save_dir = '../images-pdfs'
    index = True 
    
    @property
    @abstractmethod
    def sigfigs(self) -> int:
        "number of sigfigs to put in table"
        pass
    
    def run(self, loud=True):
        table_string = self.data_df.to_latex(index=self.index, float_format=f"%.{self.sigfigs}f")
        save_loc = f"{self.save_dir}/{self.section}/{self.name}"
        if loud:
            print(table_string)
        
        with open(f'{save_loc}.tex', 'w') as f:
            print(table_string, file=f)        
            
        with open(f'{save_loc}.json', 'w') as f:
            json.dump({'date_updated': datetime.now().strftime("%m/%d/%Y,%H:%M:%S"),
                         'name': self.name,
                         'description':self.description}, f)
        

