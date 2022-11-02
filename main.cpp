// OpenCL based simple sphere path tracer by Sam Lapere, 2016
// based on smallpt by Kevin Beason 
// http://raytracey.blogspot.com 





#include <chrono>
#include <thread>
#include "OCL.h"





Settings settings = { 720, 480, 5, 10 };

OCL ocl(settings);


void loop(){
    ocl.load2Gpu();
    ocl.render();
}



int main(int argc, char** argv){
	bool info = false;


	// console arguments
	std::vector<std::string> args;
	std::copy(argv + 1, argv + argc, std::back_inserter(args));

	if (args.size() > 0){
		if (args[0] == "info") info = true;
		else{
			cout << "Invalid argument! Valid arguments are: info" << endl;
			return 0;
		}
	}



    // initialise OpenCL
    ocl.initOpenCL(info);
    ocl.initSceneSpheres();
    ocl.initScenePlanes();

    sf::Thread thread(&loop);
    sf::RenderWindow window(sf::VideoMode(settings.image_width, settings.image_height), "Path Tracer");
    sf::Texture texture;
    texture.create(settings.image_width, settings.image_height);
    sf::Sprite sprite(texture);


    sf::Text text;

    sf::Font font;
    if (!font.loadFromFile("arial.ttf"))
    {
        // error...
    }

// select the font
    text.setFont(font); // font is a sf::Font
// set the string to display
    text.setString("Path tracer");
// set the character size
    text.setCharacterSize(18); // in pixels, not points!
// set the color
    text.setFillColor(sf::Color::Black);
// set the text style
    text.setStyle(sf::Text::Bold);

    text.setPosition(20, 20);


    sf::Text samples_text;
    samples_text.setString("Samples: " + std::to_string(settings.samples));
    samples_text.setFont(font);
    samples_text.setCharacterSize(18);
    samples_text.setFillColor(sf::Color::Black);
    samples_text.setStyle(sf::Text::Bold);
    samples_text.setPosition(20, 40);

    sf::Text bounces_text;
    bounces_text.setString("Bounces: " + std::to_string(settings.max_depth));
    bounces_text.setFont(font);
    bounces_text.setCharacterSize(18);
    bounces_text.setFillColor(sf::Color::Black);
    bounces_text.setStyle(sf::Text::Bold);
    bounces_text.setPosition(20, 60);

    sf::Text fps_text;
    fps_text.setString("FPS: ");
    fps_text.setFont(font);
    fps_text.setCharacterSize(18);
    fps_text.setFillColor(sf::Color::Black);
    fps_text.setStyle(sf::Text::Bold);
    fps_text.setPosition(20, 80);

    sf::Text time_text;
    time_text.setString("Time: ");
    time_text.setFont(font);
    time_text.setCharacterSize(18);
    time_text.setFillColor(sf::Color::Black);
    time_text.setStyle(sf::Text::Bold);
    time_text.setPosition(20, 100);


    sf::RectangleShape rect;
    rect.setSize(sf::Vector2f(150, 150));
    rect.setFillColor(sf::Color(100, 100, 100, 180));
    rect.setPosition(10, 10);



    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            //check keyboard input
            if (event.type == sf::Event::KeyPressed){
                if (event.key.code == sf::Keyboard::Left){
                    settings.samples -= 1;
                    if (settings.samples < 1) settings.samples = 1;
                    ocl.setSamples(settings.samples);
                    samples_text.setString("Samples: " + std::to_string(settings.samples));
                }else if (event.key.code == sf::Keyboard::Right){
                    settings.samples += 1;
                    ocl.setSamples(settings.samples);
                    samples_text.setString("Samples: " + std::to_string(settings.samples));
                }else if (event.key.code == sf::Keyboard::Up){
                    settings.max_depth += 1;
                    ocl.setBounces(settings.max_depth);
                    bounces_text.setString("Bounces: " + std::to_string(settings.max_depth));
                }else if (event.key.code == sf::Keyboard::Down){
                    settings.max_depth -= 1;
                    if (settings.max_depth < 1) settings.max_depth= 1;
                    ocl.setBounces(settings.max_depth);
                    bounces_text.setString("Bounces: " + std::to_string(settings.max_depth));

                }
            }
            if (event.type == sf::Event::Closed)
                window.close();
        }


        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        thread.launch();
        thread.wait();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
        auto fps = 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        fps_text.setString("FPS: " + std::to_string(fps).substr(0, 5));
        time_text.setString("Time: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()).substr(0, 5) + "ms");

        auto pixels = ocl.saveToArray();
        texture.update(pixels);

        delete[] pixels;

        window.clear();
        window.draw(sprite);
        window.draw(rect);
        window.draw(text);
        window.draw(samples_text);
        window.draw(bounces_text);
        window.draw(fps_text);
        window.draw(time_text);
        window.display();
    }

    return 0;



}

