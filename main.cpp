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

const auto model_one = fdeep::load_model("../model/one.json", true, fdeep::dev_null_logger);
const auto model_one_two = fdeep::load_model("../model/one_two.json", true, fdeep::dev_null_logger);
const auto model_first = fdeep::load_model("../model/first.json", true, fdeep::dev_null_logger);
const auto model_second = fdeep::load_model("../model/second.json", true, fdeep::dev_null_logger);

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

std::pair<std::vector<pixel>, borders> bfs(const image& img, pixel p, std::unordered_set<pixel, pixel::hash>& visited)
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

bool is_dot(const image& img)
{
    return (img.rows() < 20 && img.cols() < 20);
}

void glue(std::pair<borders,image>& component, std::pair<borders,image>& dot)
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

std::string get_components(const image& img, int line_num, bool use_dictionary)
{
    fs::create_directory("../out/" + std::to_string(line_num));

    std::unordered_set<pixel, pixel::hash> visited;
    std::vector<std::pair<borders,image>> components;
    std::vector<std::pair<borders,image>> dots;
    std::vector<std::pair<borders,image>> all_components;


    for (int j = 0; j < img.rows(); ++j) {
        for (int i = 0; i < img.cols(); ++i) {
            pixel p{j,i};
            // if pixel is black and not yet visited
            if (img.check_color(p, Color::black) && visited.find(p) == visited.cend()) {
                auto [pixels, b] = bfs(img, p, visited);
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

    for (auto component_pair : all_components) {

        bool dot = false;

        if (is_dot(component_pair.second)) {
            dot = true;

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
                    dot = false;
                }
            }
        }

        if (dot) {
            dots.emplace_back(component_pair);
        }
        else {
            components.emplace_back(component_pair);
        }
    }

    std::sort(components.begin(), components.end(), [](const auto& lhs, const auto& rhs) 
            { return (lhs.first.left() + lhs.first.right()) < (rhs.first.left() + rhs.first.right()); });

    for (auto& dot : dots) {
        int dot_middle = (dot.first.left() + dot.first.right()) / 2;
        auto component = std::min_element(components.begin(), components.end(), [&](auto const& lhs, auto const& rhs){
                return std::abs(dot_middle - lhs.first.dot_point()) < std::abs(dot_middle - rhs.first.dot_point());
        });

        if (component == components.end()) {
            std::cout << ".";
        }
        else {
            glue(*component, dot);
        }
    }

    int height_sum = 0;
    int width_sum = 0;
    for (const auto& component : components) {
        height_sum += component.second.rows();
        width_sum += component.second.cols();
    }
    double height_avg = (double)height_sum/(double)components.size();
    double width_avg = (double)width_sum/(double)components.size();

    int space_sum = 0;
    std::vector<int> space_diff;
    for (int i = 0; i < (int)components.size()-1; i++) {
        space_sum += std::min((int)(3*width_avg), std::max(components[i+1].first.left() - components[i].first.right(), 0));
        space_diff.push_back(std::max(components[i+1].first.left() - components[i].first.right(), 0));
    }

    double space_avg = 0;
    if (components.size() > 1) {
        space_avg = (double)space_sum/(double)(components.size()-1);
    }


    std::string line;
    int image_count = 0;
    std::string default_word;
    std::vector<std::string> words_candidates(1,"");

    for (int i = 0; i < (int)components.size(); i++) {
        components[i].second.save("../out/" + std::to_string(line_num) + "/" + std::to_string(image_count) + ".png");
        ++image_count;

        bool not_above_height_avg = components[i].second.rows() <= height_avg;
        auto prediction = recognize(components[i].second);


        std::vector<std::string> words_candidates_to_add;
        for (auto& word : words_candidates) {
            std::string word_append;

            if (!prediction.second.empty()) {
                word_append = word + prediction.second;
                words_candidates_to_add.push_back(word + prediction.first);
            }
            else {
                word_append = word + prediction.first;
            }

            default_word = word + prediction.first;

            if (prediction.first == "l" && not_above_height_avg) {
                words_candidates_to_add.push_back(word + "c");
                words_candidates_to_add.push_back(word + "e");
            }
            if (prediction.first == "j" && not_above_height_avg) {
                words_candidates_to_add.push_back(word + "i");
            }
            if (prediction.first == "p" && not_above_height_avg) {
                words_candidates_to_add.push_back(word + "e");
            }

            word = word_append;
        }

        for (auto& word : words_candidates_to_add) {
            words_candidates.push_back(std::move(word));
        }

        if (i != (int)components.size()-1) {
            int this_diff = std::max(components[i+1].first.left() - components[i].first.right(), 0);

            int prev_diff = (i > 0) ? std::max(components[i].first.left() - components[i-1].first.right(), 0) : 1.5*space_avg;
            int next_diff = (i+2 < (int)components.size()) ?  std::max(components[i+2].first.left() - components[i+1].first.right(), 0) : 1.5*space_avg;
            if ((this_diff >= 1.5*prev_diff || this_diff >= 1.5*next_diff) && this_diff > 1.1*width_avg && this_diff >= 1.5*space_avg) {
                if (words_candidates.size() == 1) {
                    line += words_candidates[0];
                }
                else {
                    bool found = false;
                    for (const auto& word : words_candidates) {
                        if (dictionary.find(word) != dictionary.end() && !(word.size() == 1 && word != "a")) {
                            line += word;
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        line += default_word;
                    }
                }

                // if (words_candidates.size() > 1) {
                //     line += "{";
                // }
                // for (int w = 0; w < words_candidates.size(); w++) {
                //     line +=  words_candidates[w];
                //     if (w != words_candidates.size()-1) {
                //         line += ",";
                //     }
                // }
                // if (words_candidates.size() > 1) {
                //     line += "}";
                // }

                line += " ";
                default_word.clear();
                words_candidates.clear();
                words_candidates.push_back("");
            }
        }
    }

    if (!words_candidates.empty() && !words_candidates[0].empty()) {
         // if (words_candidates.size() > 1) {
         //     line += "{";
         // }
         // for (int w = 0; w < words_candidates.size(); w++) {
         //     line +=  words_candidates[w];
         //     if (w != words_candidates.size()-1) {
         //         line += ",";
         //     }
         // }
         // if (words_candidates.size() > 1) {
         //     line += "}";
         // }
        
        
         if (words_candidates.size() == 1) {
             line += words_candidates[0];
         }
         else {
             bool found = false;
             for (const auto& word : words_candidates) {
                 if (dictionary.find(word) != dictionary.end() && !(word.size() == 1 && word != "a")) {
                     line += word;
                     found = true;
                     break;
                 }
             }

             if (!found) {
                line += default_word;
             }
         }
    }

    return line;
}

bool bfs(const image& img, pixel p, int line_num, int line_num_components, bool use_dictionary)
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
                if ((img.check_color(neighbour, Color::black) || img.check_color(neighbour, Color::white)) && visited.find(neighbour) == visited.cend()) {
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
        line.save("../out/line" + std::to_string(line_num) + ".png");
        if (line_num_components == -1 || line_num == line_num_components) {
            auto line_text = get_components(line, line_num, use_dictionary);
            if (use_dictionary) {
                auto line_fixed = corrector.FixFragment(std::wstring(line_text.begin(), line_text.end()));
                std::cout << std::string(line_fixed.begin(), line_fixed.end()) << std::endl;
            }
            else {
                std::cout << line_text << std::endl;
            }
        }
        return true;
    }

    return false;
}

void bfs(const image& img, int line_num_components, bool use_dictionary)
{
    int line_num = 0;
    for (int j = 1; j < img.rows(); ++j) {
        if (j == 1 || img.check_color({j-1,0}, Color::gray)) {
            if (bfs(img, {j,0}, line_num, line_num_components, use_dictionary)) {
                ++line_num;
            }
        }
    }
}

void process_image(const fs::path& img_name, int line_num_components, bool use_dictionary)
{
    std::cout << img_name << std::endl;
    try {
        image img(img_name);
        strips s(img);
        s.connect();
        auto strips_image = s.concatenate_strips();
        auto strips_contour = s.concatenate_strips_with_lines(false);
        auto strips_contour_original = s.concatenate_strips_with_lines(true);
        auto result_image = s.result();

        result_image.save(std::string("../out/") + std::string(img_name.filename()));
        bfs(result_image, line_num_components, use_dictionary);
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
        std::cout << "./keras2cpp path_to_img [line_num_components] [dictionary]" << std::endl;
        return -1;
    }

    if (fs::exists("../out")) {
        fs::remove_all("../out");
    }
    fs::create_directory("../out");

    int line_num_components = -1;
    bool use_dictionary = false;

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "dictionary") {
            use_dictionary = true;
        }
        else {
            line_num_components = std::stoi(argv[i]);
        }
    }

    if (use_dictionary) {
        corrector.LoadLangModel("../model/en.bin");
        std::cout << "model loaded" << std::endl;
    }

    load_dictionary();
    process_image(argv[1], line_num_components, use_dictionary);

    return 0;
}
