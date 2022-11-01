#include <SFML/Graphics.hpp>
#include <iostream>


float col = 255;




void checkEvents(sf::RenderWindow &window, sf::Time &time) {
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
    {
        col -= time.asMilliseconds() / 10.0f;
        std:: cout << col << std::endl;
    }
}

void updateTexture(sf::Texture &texture, sf::Time &time) {
    texture.create(200, 200);
    sf::Uint8* pixels = new sf::Uint8[200 * 200 * 4];
    for (int i = 0; i < 200 * 200 * 4; i += 4)
    {
        pixels[i] = (int)col;
        pixels[i + 1] = (int)col * (int)col;
        pixels[i + 2] = 0;
        pixels[i + 3] = 255;
    }
    texture.update(pixels);
    delete[] pixels;
}


int main()
{
    sf::RenderWindow window(sf::VideoMode(200, 200), "SFML works!");

   sf::Texture texture;

    
    sf::Clock clock;
    while (window.isOpen())
    {

        sf::Time elapsed = clock.restart();
        checkEvents(window, elapsed);


        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();

        updateTexture(texture, elapsed);
        sf::Sprite sprite(texture);
        window.draw(sprite);
        
        window.display();
    }

    return 0;
}