import numpy as np
class KNearestNeihgbor:


    # k to liczba najbliższych sąsiadów
    def __init__( self, k ):

        self.k = k      
        pass


    def Train( self, Dane, EtykietyDanych ):

        self.Dane = Dane
        self.EtykietyDanych = EtykietyDanych
        pass


    # Znajdź etykiety do k najlbiższych sąsiadów - to prawdopodobnie jest klasa danej testowej
    def Predict( self, DanaTestowa ):

        # Najpierw musimy policzyć odległości - to jest kłopotliwy krok w metodzie kNN
        dystanse = self.ComputeDistances( DanaTestowa )

        # Teraz musimy znaleźć najbliższych sąsiadów, a w zasadzie interesują nas ich etykiety

        # argsort sortuje, ale zwraca indeksy (a nie same wartości) najmniejszych obiektów
        indeksy_najbliższych_sąsiadów = np.argsort( dystanse )

        # znajdźmy jakie są to etykiety
        k_najbliższych_klas = self.EtykietyDanych[ indeksy_najbliższych_sąsiadów[ : self.k ] ].astype( int )

        # w końcu szukamy dominującej klasy w zbiorze k etykiet
        klasa_danej_testowej = np.argmax( np.bincount( k_najbliższych_klas ) )

        return klasa_danej_testowej
    



    # Policz odległości wg pewnej miary odległości właściwej dla tych danych
    def ComputeDistances( self, DanaTestowa ):

        # self.Dane to tablica (array) o liczbie wierszych == liczbie danych
        # rozmiar każdej danej to liczba kolumn
        liczba_danych = self.Dane.shape[ 0 ]
        wymiar_danych = self.Dane.shape[ 1 ]

        # wymiary muszą się zgadzać
        assert( DanaTestowa.shape[ 0 ] == wymiar_danych )

        #dist = [ np.linalg.norm( self.Dane[ i ] - DanaTestowa ) for i in range( liczba_danych ) ]
        #return np.array( dist )
        return np.linalg.norm( self.Dane - DanaTestowa, axis = 1 )        # ord = 2