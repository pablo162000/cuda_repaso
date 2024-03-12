#include <iostream>
#include <chrono>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <fmt/core.h>

#include <cuda_runtime.h>

namespace ch = std::chrono;
double last_time = 0;

static const std::string filename1 = "C:/Programacion Paralela/correccion_cuda/im1.jpg";
static const std::string filename2 = "C:/Programacion Paralela/correccion_cuda/im2.jpg";


static int image_width;
static int image_height;
static int image_channels;

const sf::Uint8 * host_src_image2_pixels = nullptr;
const sf::Uint8 * host_src_image1_pixels = nullptr;
 sf::Uint8 * host_src_sum_pixels = nullptr;

sf::Uint8 * device_src_image2_pixels = nullptr;
sf::Uint8 * device_src_image1_pixels = nullptr;


extern "C" void kernel_suma(unsigned char* src_image, unsigned char* dst_image, int width, int height);

#define CHECK(expr) {                       \
        auto err = (expr);                  \
        if (err != cudaSuccess) {           \
            printf("%d: %s in % s at line % d\n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);             \
        }                                   \
    }

void reset_texture(sf::Texture &texture) {
    texture.update(host_src_image2_pixels);
    last_time = 0;
}

void convert_sum_image() {
    size_t buffer_size = image_width*image_height*image_channels;

    //copiar imagen host==>device
    CHECK(cudaMemcpy(device_src_image2_pixels, host_src_image2_pixels, buffer_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_src_image1_pixels, host_src_image1_pixels, buffer_size, cudaMemcpyHostToDevice));

    kernel_suma(device_src_image2_pixels,device_src_image1_pixels,image_width,image_height);

    CHECK(cudaGetLastError());

    //copiar imagen device==>host
    CHECK(cudaMemcpy(host_src_sum_pixels, device_src_image1_pixels, buffer_size, cudaMemcpyDeviceToHost));
}

int main() {

    sf::Text text;
    sf::Font font;
    {
        font.loadFromFile("arial.ttf");
        text.setFont(font);
        text.setString("Mandelbrot set");
        text.setCharacterSize(24); // in pixels, not points!
        text.setFillColor(sf::Color::White);
        text.setStyle(sf::Text::Bold);
        text.setPosition(10,10);
    }

    sf::Text textOptions;
    {
        font.loadFromFile("arial.ttf");
        textOptions.setFont(font);
        textOptions.setCharacterSize(24);
        textOptions.setFillColor(sf::Color::White);
        textOptions.setStyle(sf::Text::Bold);
        textOptions.setString("OPTIONS: [R] Reset [B] Blur [Up|Down] Change matrix size ");
    }

    //load image
    sf::Image im;
    im.loadFromFile(filename1);

    host_src_image1_pixels= im.getPixelsPtr();


    sf::Image im2;
    im2.loadFromFile(filename2);

    host_src_image2_pixels= im2.getPixelsPtr();
    //host_src_image_pixels = im.getPixelsPtr();

    if(host_src_image1_pixels){
        fmt::println("se cargo la image {}",1);
    }

    image_width = im.getSize().x;
    image_height = im.getSize().y;
    image_channels = 4;

    //--inicializar buffers
    size_t buffer_size = image_width*image_height*image_channels;

    host_src_sum_pixels = (sf::Uint8 *)malloc(buffer_size);


    CHECK(cudaMalloc(&device_src_image1_pixels, buffer_size));
    CHECK(cudaMalloc(&device_src_image2_pixels, buffer_size));

    //--

    sf::Texture texture;
    texture.create(image_width, image_height);
    texture.update(im.getPixelsPtr());

    int w = 1600;
    int h = 900;

    sf::RenderWindow  window(sf::VideoMode(w, h), "CUDA sum example");


    sf::Sprite sprite;
    {
        sprite.setTexture(texture);

        float scaleFactorX = w * 1.0 / im.getSize().x;
        float scaleFactorY = h * 1.0 / im.getSize().y;
        sprite.scale(scaleFactorX, scaleFactorY);
    }

    sf::Clock clock;

    sf::Clock clockFrames;
    int frames = 0;
    int fps = 0;
    bool paused = false;

    textOptions.setPosition(10, window.getView().getSize().y-40);
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if(event.type==sf::Event::KeyReleased) {
                if(event.key.scancode==sf::Keyboard::Scan::R) {
                    reset_texture(texture);
                }
                else if(event.key.scancode==sf::Keyboard::Scan::B) {
                    auto start = ch::high_resolution_clock::now();
                    {
                        convert_sum_image();
                        auto end = ch::high_resolution_clock::now();
                        ch::duration<double, std::milli> tiempo = end - start;
                        last_time = tiempo.count();
                    }

                    texture.update(host_src_sum_pixels);
                }

            }else
             if(event.type==sf::Event::Resized) {
                float scaleFactorX = event.size.width *1.0 / im.getSize().x;
                float scaleFactorY = event.size.height *1.0 /im.getSize().y;

                sprite = sf::Sprite();
                sprite.setTexture(texture);
                sprite.scale(scaleFactorX, scaleFactorY);
            }
        }

        auto msg = fmt::format(" suma time: {}ms, FPS: {}", last_time, fps);
        text.setString(msg);

        if(clockFrames.getElapsedTime().asSeconds()>=1.0) {
            fps = frames;
            frames = 0;
            clockFrames.restart();
        }
        frames++;

        window.clear(sf::Color::Black);
        window.draw(sprite);
        window.draw(text);
        window.draw(textOptions);
        window.display();
    }


    CHECK(cudaFree(device_src_image1_pixels));
    CHECK(cudaFree(device_src_image2_pixels));
    free(host_src_sum_pixels);

    return 0;
}
