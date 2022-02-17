class BaseWidget():
    
    def __init__(self):
        pass
    
    def display(self):
        """Display in streamlit application

        Raises:
            NotImplementedError: forces children classes to implement this
        """
        raise NotImplementedError
