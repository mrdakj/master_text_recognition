#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <fdeep/fdeep.hpp>
#include "include/image.h"
#include "include/line_segmentation.h"
#include "jamspell/spell_corrector.hpp"
#include <cassert>
#include <optional>

const auto model_one = fdeep::load_model("../model/one.json", true, fdeep::dev_null_logger);
const auto model_one_two = fdeep::load_model("../model/one_two.json", true, fdeep::dev_null_logger);
const auto model_first = fdeep::load_model("../model/first.json", true, fdeep::dev_null_logger);
const auto model_second = fdeep::load_model("../model/second.json", true, fdeep::dev_null_logger);

void glue(std::pair<borders,image> & component, std::pair<borders,image> & dot);

NJamSpell::TSpellCorrector corrector;

std::unordered_set<std::string> dictionary;

void load_dictionary()
{
    std::ifstream infile("../dictionary/google-10000-english.txt");
    std::string word;
    while (std::getline(infile, word))
    {
        dictionary.insert(word);
    }
    std::cout << "loading dictionary done" << std::endl;
}

std::pair<std::string, std::string> recognize(image& component);

std::pair<std::vector<pixel>, borders> bfs_get_components(const image& img, pixel p, std::unordered_set<pixel, pixel::hash>& visited)
{
    std::queue<pixel> q;
    q.push(p);
    visited.insert(p);

    std::vector<pixel> black;

    // left, right, top, bottm
    borders b{p.i, p.i, p.j, p.j};

    while (!q.empty()) {
        auto current_pixel = q.front();
        black.push_back(current_pixel);
        q.pop();
        b.update(current_pixel);

        for (int dj : {-1, 0, 1}) {
            for (int di : {-1, 0, 1}) {
                pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                // if neighbour is black and not yet visited
                if (img.check_color(neighbour, Color::black) && visited.find(neighbour) == visited.cend()) {
                    q.push(neighbour);
                    visited.insert(neighbour);
                }
            }
        }
    }

    return {black, b};
}

bool is_dot_candidate(const image& img)
{
    return img.rows() < 20 && img.cols() < 20;
}

bool is_dot(const std::pair<borders,image> & component_pair, const std::vector<std::pair<borders,image>> & all_components)
{
    if (!is_dot_candidate(component_pair.second)) return false;

    int dot_middle = (component_pair.first.left() + component_pair.first.right()) / 2;
    int min_distance = 999999;
    int component_index = 0;

    for (int component_i = 0; component_i < all_components.size(); ++component_i) {
        auto& other_pair = all_components[component_i];
        if (other_pair.first == component_pair.first) {
            continue;
        }

        int distance = std::abs(dot_middle - other_pair.first.dot_point());
        if (distance < min_distance) {
            min_distance = distance;
            component_index = component_i;
        }
    }

    auto component_copy = all_components[component_index];
    auto component_pair_copy = component_pair;

    glue(component_copy, component_pair_copy);
    auto prediction = recognize(component_copy.second);
    if (prediction.first != "i" && prediction.first != "j") {
        if (prediction.second.empty() || 
            (prediction.second[0] != 'i' && prediction.second[1] != 'i' &&
             prediction.second[0] != 'j' && prediction.second[1] != 'j')) {
            return false;
        }
    }

    return true;
}


std::pair<std::vector<std::pair<borders,image>>, std::vector<std::pair<borders,image>>> get_components_and_dots(const std::vector<std::pair<borders,image>> & all_components)
{
    std::vector<std::pair<borders,image>> components;
    std::vector<std::pair<borders,image>> dots;
    for (auto component_pair : all_components) {
        if (is_dot(component_pair, all_components)) {
            dots.emplace_back(component_pair);
        }
        else {
            components.emplace_back(component_pair);
        }
    }

    return {components, dots};
}

void glue(std::pair<borders,image> & component, std::pair<borders,image> & dot)
{
    component.second.add_border(Direction::right, dot.first.right() - component.first.right());
    component.second.add_border(Direction::left, component.first.left() - dot.first.left());
    dot.second.add_border(Direction::right,  component.first.right() - dot.first.right());
    dot.second.add_border(Direction::left,  dot.first.left() - component.first.left());

    component.second.add_border(Direction::up, component.first.top() - dot.first.bottom());

    component.second = component.second.concatenate_horizontal(dot.second);

    component.first = borders(
            std::min(component.first.left(), dot.first.left()),
            std::max(component.first.right(), dot.first.right()),
            dot.first.top(),
            component.first.bottom(),
            component.first.dot_point());
}

std::pair<double, double> get_avg_width_height(const std::vector<std::pair<borders,image>>& components)
{
    int height_sum = 0;
    int width_sum = 0;
    for (const auto& component : components) {
        height_sum += component.second.rows();
        width_sum += component.second.cols();
    }
    double height_avg = (double)height_sum/(double)components.size();
    double width_avg = (double)width_sum/(double)components.size();

    return {width_avg, height_avg};
}


std::pair<int, double> get_space_sum_and_avg(const std::vector<std::pair<borders,image>>& components, double width_avg)
{
    int space_sum = 0;
    for (int i = 0; i < (int)components.size()-1; i++) {
        space_sum += std::min((int)(3*width_avg), std::max(components[i+1].first.left() - components[i].first.right(), 0));
    }

    double space_avg = 0;
    if (components.size() > 1) {
        space_avg = (double)space_sum/(double)(components.size()-1);
    }

    return {space_sum, space_avg};
}


std::optional<std::string> select_word(const std::vector<std::string>& word_candidates)
{
    for (const auto& word : word_candidates) {
        if (dictionary.find(word) != dictionary.end() && !(word.size() == 1 && word != "a")) {
            return std::make_optional(word);
        }
    }

    return std::nullopt;
}

void update_candidates_and_default_word(std::vector<std::string> & word_candidates, std::string & default_word, image& img, double height_avg)
{
    bool above_avg_height = img.rows() > height_avg;
    auto prediction = recognize(img);

    std::vector<std::string> words_candidates_to_add;
    for (auto& word : word_candidates) {
        std::string word_append;

        if (!prediction.second.empty()) {
            word_append = word + prediction.second;
            words_candidates_to_add.push_back(word + prediction.first);
        }
        else {
            word_append = word + prediction.first;
        }

        default_word = word + prediction.first;

        if (prediction.first == "l" && !above_avg_height) {
            words_candidates_to_add.push_back(word + "c");
            words_candidates_to_add.push_back(word + "e");
        }
        if (prediction.first == "j" && !above_avg_height) {
            words_candidates_to_add.push_back(word + "i");
        }
        if (prediction.first == "p" && !above_avg_height) {
            words_candidates_to_add.push_back(word + "e");
        }

        word = word_append;
    }

    for (auto& word : words_candidates_to_add) {
        word_candidates.push_back(std::move(word));
    }
}

void merge_dots_and_components(std::vector<std::pair<borders,image>> & dots, std::vector<std::pair<borders,image>> & components)
{
    for (auto& dot : dots) {
        int dot_middle = (dot.first.left() + dot.first.right()) / 2;
        auto component = std::min_element(components.begin(), components.end(), [&](auto const& lhs, auto const& rhs){
                return std::abs(dot_middle - lhs.first.dot_point()) < std::abs(dot_middle - rhs.first.dot_point());
        });

        glue(*component, dot);
    }
}

void sort_components(std::vector<std::pair<borders,image>> & components)
{
    std::sort(components.begin(), components.end(), [](const auto& lhs, const auto& rhs) {
            return (lhs.first.left() + lhs.first.right()) < (rhs.first.left() + rhs.first.right());
    });
}

std::vector<std::pair<borders,image>> get_components(const image& img)
{
    std::vector<std::pair<borders,image>> all_components;
    std::unordered_set<pixel, pixel::hash> visited;

    for (int j = 0; j < img.rows(); ++j) {
        for (int i = 0; i < img.cols(); ++i) {
            pixel p{j,i};
            // if pixel is black and not yet visited
            if (img.check_color(p, Color::black) && visited.find(p) == visited.cend()) {
                auto [pixels, b] = bfs_get_components(img, p, visited);
                image im(b.height(), b.width());
                for (auto pp : pixels) {
                    im(pp.j - b.top(), pp.i - b.left()) = img(pp);
                }

                int min_j = 0;
                int max_j = 0;
                for (int j_pom = 0; j_pom < im.cols(); ++j_pom) {
                    for (int i_half = 0; i_half < im.rows()/2; ++i_half) {
                        if (im.check_color({i_half, j_pom}, Color::black)) {
                            if (min_j == 0) {
                                min_j = j_pom;
                            }
                            max_j = j_pom;
                        }
                    }
                }

                int dot_point = b.left() + (min_j + max_j)/2;
                b.set_dot_point(dot_point);

                all_components.emplace_back(b, std::move(im));

            }
        }
    }

    return all_components;
}

std::string get_line_text(const image& img, int line_num)
{
    // fs::create_directory("../out/" + std::to_string(line_num));

    auto all_components = get_components(img);

    auto [components, dots] = get_components_and_dots(all_components);
    merge_dots_and_components(dots, components);
    sort_components(components);

    auto [width_avg, height_avg] = get_avg_width_height(components);
    auto [space_sum, space_avg] = get_space_sum_and_avg(components, width_avg);

    std::string line;
    // int image_count = 0;
    std::string default_word;
    std::vector<std::string> words_candidates = {""};

    for (int i = 0; i < (int)components.size(); i++) {
        // components[i].second.save("../out/" + std::to_string(line_num) + "/" + std::to_string(image_count) + ".png");
        // ++image_count;

        update_candidates_and_default_word(words_candidates, default_word, components[i].second, height_avg);

        auto is_word_end = [&](int i) {
            int this_diff = std::min((int)(3*width_avg), 
                                     std::max(components[i+1].first.left() - components[i].first.right(), 0));

            int prev_diff = (i > 0) ? std::min((int)(3*width_avg), 
                                                std::max(components[i].first.left() - components[i-1].first.right(), 0)) 
                                    : 1.5*space_avg;

            int next_diff = (i+2 < (int)components.size()) ?  std::min((int)(3*width_avg), 
                                                              std::max(components[i+2].first.left() - components[i+1].first.right(), 0)) 
                                                           : 1.5*space_avg;

            return (this_diff >= 1.5*prev_diff || this_diff >= 1.5*next_diff) && 
                   this_diff > 1.1*width_avg && 
                   this_diff >= 1.5*space_avg;
        };

        if (i == (int)components.size()-1 || is_word_end(i)) {
            auto selected_word = select_word(words_candidates);
            line += (selected_word) ? *selected_word : default_word;

            if (i != (int)components.size()-1) {
                line += " ";
                default_word = "";
                words_candidates = {""};
            }
        }
    }

    return line;
}

std::optional<image> bfs_segment_line(const image& img, pixel p)
{
    std::unordered_set<pixel, pixel::hash> visited;
    std::queue<pixel> q;
    q.push(p);
    visited.insert(p);

    int black_count = 0;

    // left, right, top, bottm
    borders b{p.i, p.i, p.j, p.j};

    while (!q.empty()) {
        auto current_pixel = q.front();
        q.pop();
        b.update(current_pixel);

        for (int dj : {-1, 0, 1}) {
            for (int di : {-1, 0, 1}) {
                if (std::abs(dj) == std::abs(di)) {
                    continue;
                }

                pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                if ((img.check_color(neighbour, Color::black) || img.check_color(neighbour, Color::white)) && 
                    visited.find(neighbour) == visited.cend()) {
                    if (img.check_color(neighbour, Color::black)) {
                        ++black_count;
                    }
                    q.push(neighbour);
                    visited.insert(neighbour);
                }
            }
        }
    }

    if (black_count > 0) {
        image line(b.height(), b.width());
        for (auto pp : visited) {
            line(pp.j - b.top(), pp.i - b.left()) = img(pp);
        }
        line.crop();
        return std::make_optional(line);
    }

    return std::nullopt;
}

void get_lines_text(const image& img, bool use_dictionary, std::ofstream & out_file)
{
    int line_num = 0;
    for (int j = 1; j < img.rows(); ++j) {
        if (j == 1 || img.check_color({j-1,0}, Color::gray)) {
            auto maybe_line = bfs_segment_line(img, {j,0});

            if (maybe_line) {
                // maybe_line->save("../out/line" + std::to_string(line_num) + ".png");
                auto line_text = get_line_text(*maybe_line, line_num);

                if (use_dictionary) {
                    auto line_fixed = corrector.FixFragment(std::wstring(line_text.begin(), line_text.end()));
                    std::cout << std::string(line_fixed.begin(), line_fixed.end()) << std::endl;
                    out_file << std::string(line_fixed.begin(), line_fixed.end()) << std::endl;
                }
                else {
                    std::cout << line_text << std::endl;
                    out_file << line_text << std::endl;
                }

                ++line_num;
            }
        }
    }
}

void process_image(const fs::path& img_name, bool use_dictionary)
{
    std::ofstream out_file("../out/" + img_name.stem().string() + ((use_dictionary) ? "_dictionary.txt" : ".txt"));

    std::cout << img_name << std::endl;
    try {
        image img(img_name);
        strips s(img);
        s.connect();
        auto strips_image = s.concatenate_strips();
        auto strips_contour = s.concatenate_strips_with_lines(false);
        auto strips_contour_original = s.concatenate_strips_with_lines(true);
        auto result_image = s.result();

        // result_image.save(std::string("../out/") + std::string(img_name.filename()));
        get_lines_text(result_image, use_dictionary, out_file);
    } 
    catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }
}

std::pair<std::string, std::string> recognize(image& component)
{
    component.resize(28,28);
    auto t = component.get_tensor(0,1);
    std::string character =  std::string(1, char(model_one.predict_class({t}) + 97));

    if (model_one_two.predict_class({t}) == 1) {
        // we have two letters maybe
        return {character, std::string(1, char(model_first.predict_class({t}) + 97)) + std::string(1, char(model_second.predict_class({t}) + 97))};
    }

    return {character, ""};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "./keras2cpp path_to_img [dictionary]" << std::endl;
        return -1;
    }

    fs::create_directory("../out");

    if (fs::is_directory(argv[1])) {
        corrector.LoadLangModel("../model/en.bin");
        std::cout << "model loaded" << std::endl;
        load_dictionary();

        for (auto it = fs::directory_iterator(argv[1]); it != fs::directory_iterator(); ++it) {
            if (fs::path(*it).extension() == ".png") {
                process_image(*it, false);
                process_image(*it, true);
            }
        }
    }
    else {
        bool use_dictionary = false;

        if (argc == 3 && std::string(argv[2]) == "dictionary") {
            use_dictionary = true;
        }

        if (use_dictionary) {
            corrector.LoadLangModel("../model/en.bin");
            std::cout << "model loaded" << std::endl;
        }

        load_dictionary();
        process_image(argv[1], use_dictionary);
    }

    return 0;
}
