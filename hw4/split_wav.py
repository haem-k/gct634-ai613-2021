import wave

# times between which to extract the wave from
start = [208.0, 288.0, 458.7, 599.5, 778.8, 869.8, 992.0, 1072, 1200, 1276, 1384, 1462, 1504, 1605, 1772, 1952, 2066, 2183, 2316, 2396.9] # seconds
end = [286.0, 458.7, 599.0, 778.7, 869.7, 980.0, 1072, 1196.0, 1276, 1384, 1461.1, 1504, 1604.8, 1760, 1952, 2065, 2182.6, 2315.3, 2396.7, 2518] # seconds

songs = 20
song_names = ['grandpa', 'spring1', 'spring2', 'spring3', 'pelican', 'library', 'saloon', 'summer1', 'summer2', 'summer3', 'guild', 'mines', 'carpenter', 'dance', 'fall1', 'fall2', 'fall3', 'winter1', 'winter2', 'winter3']

for i in range(songs):
    # file to extract the snippet from
    with wave.open('./data/sample/stv_medley.wav', "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        # set position in wave to start of segment
        infile.setpos(int(start[i] * framerate))
        # extract data
        data = infile.readframes(int((end[i] - start[i]) * framerate))

    # write the extracted data to a new file
    with wave.open(f'./data/sample/{song_names[i]}.wav', 'w') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(framerate)
        outfile.setnframes(int(len(data) / sampwidth))
        outfile.writeframes(data)