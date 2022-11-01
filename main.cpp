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

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // initialise OpenCL
    ocl.initOpenCL(info);
    ocl.initSceneSpheres();
    ocl.initScenePlanes();

    sf::Thread thread(&loop);
    sf::RenderWindow window(sf::VideoMode(settings.image_width, settings.image_height), "Path Tracer");
    sf::Texture texture;
    texture.create(settings.image_width, settings.image_height);
    sf::Sprite sprite(texture);


    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            //check keyboard input
            if (event.type == sf::Event::KeyPressed){
                if (event.key.code == sf::Keyboard::Left){
                    ocl.animate();
                }
            }
            if (event.type == sf::Event::Closed)
                window.close();
        }

        thread.launch();
        thread.wait();

        auto pixels = ocl.saveToArray();
        texture.update(pixels);

        delete[] pixels;

        window.clear();
        window.draw(sprite);
        window.display();
    }

    return 0;


	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
}

