for i in /faces/moeimouto-faces/*
do
    echo "Processing $i"
    ruby /animeface-2009/animeface-ruby/face_collector.rb --src "$i" --dest "/faces/new-data/$(basename "$i")" --threshold 0.5 --margin 1
done