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


    std::map<std::string, sf::Text> texts;

    sf::Font font;
    if (!font.loadFromFile("arial.ttf"))
    {
        // error...
    }

    texts["name"]    = sf::Text("Path tracer",                                        font, 18);
    texts["name"].setPosition(10, 10);
    texts["samples"] = sf::Text("Samples: " + std::to_string(settings.samples),   font, 18);
    texts["samples"].setPosition(10, 30);
    texts["bounces"] = sf::Text("Bounces: " + std::to_string(settings.max_depth), font, 18);
    texts["bounces"].setPosition(10, 50);
    texts["fps"]     = sf::Text("FPS: " + std::to_string(0),                      font, 18);
    texts["fps"].setPosition(10, 70);
    texts["time"]    = sf::Text("Time: " + std::to_string(0),                     font, 18);
    texts["time"].setPosition(10, 90);

    for (auto& [key, value] : texts) {
        value.setFillColor(sf::Color::Black);
        value.setStyle(sf::Text::Bold);
    }


    sf::RectangleShape rect;
    rect.setSize(sf::Vector2f(150, 150));
    rect.setFillColor(sf::Color(100, 100, 100, 180));
    rect.setPosition(10, 10);


    int edge_index = 0;
    std::vector <std::string> edges = { "up_left", "down_right" };

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
                    texts["samples"].setString("Samples: " + std::to_string(settings.samples));
                }else if (event.key.code == sf::Keyboard::Right){
                    settings.samples += 1;
                    ocl.setSamples(settings.samples);
                    texts["samples"].setString("Samples: " + std::to_string(settings.samples));
                }else if (event.key.code == sf::Keyboard::Up){
                    settings.max_depth += 1;
                    ocl.setBounces(settings.max_depth);
                    texts["bounces"].setString("Bounces: " + std::to_string(settings.max_depth));
                }else if (event.key.code == sf::Keyboard::Down){
                    settings.max_depth -= 1;
                    if (settings.max_depth < 1) settings.max_depth= 1;
                    ocl.setBounces(settings.max_depth);
                    texts["bounces"].setString("Bounces: " + std::to_string(settings.max_depth));
                }
                else if( event.key.code == sf::Keyboard::Space ){
                    edge_index = (edge_index + 1) % 3;
                    std::cout << "edge_index: " << edges[edge_index] << std::endl;
                }

                else if( event.key.code >= sf::Keyboard::A && event.key.code <= sf::Keyboard::Z ){
                    ocl.animate(event.key.code, edges[edge_index]);
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

        texts["fps"].setString("FPS: " + std::to_string(fps).substr(0, 5));
        texts["time"].setString("Time: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()).substr(0, 5) + "ms");

        auto pixels = ocl.saveToArray();
        texture.update(pixels);
        delete[] pixels;

        window.clear();
        window.draw(sprite);
        window.draw(rect);
        for (const auto& text : texts){
            window.draw(text.second);
        }
        window.display();
    }

    return 0;



}

