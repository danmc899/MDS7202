def animales(input):
    
    '''
    Como hace el animalito?
    '''
    
    if input.lower() == 'gato':
        print('miau')
    elif input.lower() == 'perro':
        print('guau')
    elif input.lower() == 'fox':
        print('https://www.youtube.com/watch?v=jofNR_WkoCE')
    else:
        raise ValueError('animal no reconocido :(')
