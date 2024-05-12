import numpy as np
class KNearestNeihgbor:
    def __init__( self, k ):
        self.k = k      
        pass

    def Train( self, data, labels ):
        self.data = data
        self.labels = labels
        pass

    def Predict( self, data ):
        # Najpierw musimy policzyć odległości - to jest kłopotliwy krok w metodzie kNN
        distance = self.ComputeDistances( data )

        # Teraz musimy znaleźć najbliższych sąsiadów, a w zasadzie interesują nas ich etykiety

        # argsort sortuje, ale zwraca indeksy (a nie same wartości) najmniejszych obiektów
        knn_indexes = np.argsort( distance )

        # znajdźmy jakie są to etykiety
        knn_classes = self.labels[ knn_indexes[ : self.k ] ].astype( int )

        # w końcu szukamy dominującej klasy w zbiorze k etykiet
        classes = np.argmax( np.bincount( knn_classes ) )

        return 0
    
    # Policz odległości wg pewnej miary odległości właściwej dla tych danych
    def ComputeDistances( self, test_data ):

        # self.Dane to tablica (array) o liczbie wierszych == liczbie danych
        # rozmiar każdej danej to liczba kolumn
        data_count = self.data.shape[ 0 ]
        wymiar_danych = self.data.shape[ 1 ]

        # wymiary muszą się zgadzać
        assert( test_data.shape[ 0 ] == wymiar_danych )

        #dist = [ np.linalg.norm( self.Dane[ i ] - DanaTestowa ) for i in range( liczba_danych ) ]
        #return np.array( dist )
        return np.linalg.norm( self.data - test_data, axis = 1 )        # ord = 2