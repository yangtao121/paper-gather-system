class DifyWorkflowConfig:
    def __init__(self, 
                 api_url: str,
                 api_key: str,
                 user: str = "ai-server"    
                 ):
        self.api_url = api_url
        self.api_key = api_key
        self.user = user
        
        