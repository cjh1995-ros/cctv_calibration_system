from modules.cameras.projectors import UnDistorter, Distorter

# Test undistorter

if __name__ == '__main__':
    ru = 1.5
    k = [-0.2, 0.1]
    
    rd = Distorter.polynomial(ru, k)

    print(rd)
    
    new_ru = UnDistorter.polynomial(rd, k)
    print(f"Original ru: {ru}, new ru: {new_ru}")    