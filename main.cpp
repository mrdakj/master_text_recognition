#include <iostream>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <cassert>
#include <optional>
#include <limits>
#include <fdeep/fdeep.hpp>
#include "jamspell/spell_corrector.hpp"
#include "include/image.h"
#include "include/line_segmentation.h"

struct recognition_models {
    // collection of models used for letters recognition

    recognition_models()
        : model_one(fdeep::load_model("../model/one.json", true, fdeep::dev_null_logger))
        , model_one_two(fdeep::load_model("../model/one_two.json", true, fdeep::dev_null_logger))
        , model_first(fdeep::load_model("../model/first.json", true, fdeep::dev_null_logger))
        , model_second(fdeep::load_model("../model/second.json", true, fdeep::dev_null_logger))
    {
    }

    // model for single letter prediction
    // a, b, c, ..., z
    const fdeep::model model_one;
    // model for one vs two letters prediction
    // a, ..., z -> one letter
    // aa, ..., az, ..., za, ..., zz -> two letters
    const fdeep::model model_one_two;
    // model for the first bigram letter prediction
    // aa, ab, ..., az -> a
    const fdeep::model model_first;
    // model for the second bigram letter prediction
    // aa, ba, ..., za -> a
    const fdeep::model model_second;
};

class image_recognizer {
    typedef std::pair<borders,image> t_component;

public:
    image_recognizer(bool use_dictionary = true, bool debug = false)
        : m_dictionary(get_dictionary())
        , m_use_spell_check(use_dictionary)
        , m_debug(debug)
        , m_output_dir("out")
    {
        if (m_use_spell_check) {
            // load spell check model for english language
            m_corrector.LoadLangModel("../model/en.bin");
        }

        fs::create_directory(m_output_dir);
    }

    void recognize_image(const fs::path& img_name)
    {
        try {
            std::cout << img_name << std::endl;
            // set temporary vars
            set_tmp(img_name);

            if (m_debug) {
                fs::create_directory(m_debug_output_tmp);
            }

            image img(img_name);
            // segment lines
            auto result_image = segment_lines(img);

            if (m_debug) {
                result_image.save(m_debug_output_tmp / img_name.filename());
            }

            get_lines_text(result_image);
            // restore temporary vars
            clear_tmp();
        } 
        catch (std::exception& ex) {
            std::cerr << ex.what() << std::endl;
        }
    }

private:
    image segment_lines(const image& img) const
    {
        // returns image with line separating paths
        strips s(img);
        s.connect();
        auto strips_image = s.concatenate_strips();
        auto strips_contour = s.concatenate_strips_with_lines(false);
        auto strips_contour_original = s.concatenate_strips_with_lines(true);
        auto result_image = s.result();
        return result_image;
    }

    void set_tmp(const fs::path& img_name)
    {
        // set output vars for this image
        m_output_file_tmp = m_output_dir / (img_name.stem().string() + ((m_use_spell_check) ? "_dictionary.txt" : ".txt"));
        if (m_debug) {
            m_debug_output_tmp = m_output_dir / (img_name.stem().string() + "_debug");
        }
    }

    void clear_tmp()
    {
        // clear output vars
        m_output_file_tmp.clear();
        m_debug_output_tmp.clear();
    }

    std::unordered_set<std::string> get_dictionary() const
    {
        // read dictionary of english words and store it in a hash
        std::unordered_set<std::string> dictionary;
        std::ifstream infile("../dictionary/google-10000-english.txt");
        std::string word;

        while (std::getline(infile, word)) {
            dictionary.insert(word);
        }

        return dictionary;
    }

    std::pair<std::vector<pixel>, borders> bfs_get_component(const image& img, pixel p, std::unordered_set<pixel, pixel::hash>& visited) const
    {
        // get component pixels and bounding box
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

    bool is_dot_candidate(const image& img) const
    {
        return img.rows() < 20 && img.cols() < 20;
    }

    bool is_dot(const t_component& component, const std::vector<t_component> & all_components) const
    {
        if (!is_dot_candidate(component.second)) return false;

        int dot_middle = (component.first.left() + component.first.right()) / 2;
        int min_distance = std::numeric_limits<int>::max();
        int component_index = 0;

        // find the nearest component according to horizontal distance
        for (int component_i = 0; component_i < all_components.size(); ++component_i) {
            auto& other_pair = all_components[component_i];
            if (other_pair.first == component.first) {
                continue;
            }

            int distance = std::abs(dot_middle - other_pair.first.dot_point());
            if (distance < min_distance) {
                min_distance = distance;
                component_index = component_i;
            }
        }

        // copy the component and the nearest component so we can glue them together
        auto nearest_component_copy = all_components[component_index];
        auto component_copy = component;
        // add component (potential dot) to the top of the nearest component (potential part of letter i or j)
        glue(nearest_component_copy, component_copy);

        auto prediction = recognize(nearest_component_copy.second);

        return  // merged component is i
                prediction.first == "i" || 
                // merged component is j
                prediction.first == "j" ||
                // merged component is bigram that contains i or j
                (!prediction.second.empty() &&
                (prediction.second[0] == 'i' || prediction.second[1] == 'i' ||
                 prediction.second[0] == 'j' || prediction.second[1] == 'j'));
    }

    std::pair<std::vector<t_component>, std::vector<t_component>> get_components_and_dots(const std::vector<t_component> & all_components) const
    {
        // separate all components into dots and non-dot components
        std::vector<t_component> components;
        std::vector<t_component> dots;

        for (auto component : all_components) {
            if (is_dot(component, all_components)) {
                dots.emplace_back(component);
            }
            else {
                components.emplace_back(component);
            }
        }

        return {components, dots};
    }

    void glue(t_component & component, t_component & dot) const
    {
        component.second.add_border(Direction::right, dot.first.right() - component.first.right());
        component.second.add_border(Direction::left, component.first.left() - dot.first.left());
        dot.second.add_border(Direction::right,  component.first.right() - dot.first.right());
        dot.second.add_border(Direction::left,  dot.first.left() - component.first.left());

        component.second.add_border(Direction::up, component.first.top() - dot.first.bottom());

        component.second = component.second.concatenate_horizontal(dot.second);

        // update borders of the new component
        component.first = borders(
                // left
                std::min(component.first.left(), dot.first.left()),
                // right
                std::max(component.first.right(), dot.first.right()),
                // top
                dot.first.top(),
                // bottom
                component.first.bottom(),
                // dot point
                component.first.dot_point());
    }

    std::pair<double, double> get_avg_width_height(const std::vector<t_component>& components) const
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

    std::pair<int, double> get_space_sum_and_avg(const std::vector<t_component>& components, double width_avg) const
    {
        int space_sum = 0;
        for (int i = 0; i < (int)components.size()-1; ++i) {
            // space should be in interval [0, 3*average width]
            space_sum += std::min((int)(3*width_avg), 
                                   std::max(components[i+1].first.left() - components[i].first.right(), 0));
        }

        double space_avg = 0;
        if (components.size() > 1) {
            space_avg = (double)space_sum/(double)(components.size()-1);
        }

        return {space_sum, space_avg};
    }


    std::optional<std::string> select_word(const std::vector<std::string>& word_candidates) const
    {
        // choose a word that exists in the dictionary
        for (const auto& word : word_candidates) {
            if (m_dictionary.find(word) != m_dictionary.end()) {
                return std::make_optional(word);
            }
        }

        return std::nullopt;
    }

    void update_candidates_and_default_word(std::vector<std::string> & word_candidates, std::string & default_word, image& component, double height_avg) const
    {
        // update word candidates and default word with new letter(s)
        bool above_avg_height = component.rows() > height_avg;
        auto prediction = recognize(component);

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

    void merge_dots_and_components(std::vector<t_component> & dots, std::vector<t_component> & components) const
    {
        for (auto& dot : dots) {
            int dot_middle = (dot.first.left() + dot.first.right()) / 2;
            auto component = std::min_element(components.begin(), components.end(), [&](auto const& lhs, auto const& rhs){
                    return std::abs(dot_middle - lhs.first.dot_point()) < std::abs(dot_middle - rhs.first.dot_point());
            });

            // add dot to the top of the component
            glue(*component, dot);
        }
    }

    void sort_components(std::vector<t_component> & components) const
    {
        std::sort(components.begin(), components.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first.dot_point() < rhs.first.dot_point();
        });
    }

    std::vector<t_component> get_components(const image& img) const
    {
        std::vector<t_component> all_components;
        std::unordered_set<pixel, pixel::hash> visited;

        for (int j = 0; j < img.rows(); ++j) {
            for (int i = 0; i < img.cols(); ++i) {
                pixel p{j,i};
                // if pixel is black and not yet visited
                if (img.check_color(p, Color::black) && visited.find(p) == visited.cend()) {
                    auto [pixels, b] = bfs_get_component(img, p, visited);
                    image im(b.height(), b.width());
                    for (auto pp : pixels) {
                        im(pp.j - b.top(), pp.i - b.left()) = img(pp);
                    }

                    // find minimum left and maximum right point of the upper half part of the component
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

                    // dot point is the middle (horizontal) point of the upper half part of the component
                    int dot_point = b.left() + (min_j + max_j)/2;
                    b.set_dot_point(dot_point);

                    all_components.emplace_back(b, std::move(im));

                }
            }
        }

        return all_components;
    }

    std::string get_line_text(const image& img, int line_num) const
    {
        if (m_debug) {
            fs::create_directory(m_debug_output_tmp / std::to_string(line_num));
        }

        auto all_components = get_components(img);

        auto [components, dots] = get_components_and_dots(all_components);
        merge_dots_and_components(dots, components);
        sort_components(components);

        auto [width_avg, height_avg] = get_avg_width_height(components);
        auto [space_sum, space_avg] = get_space_sum_and_avg(components, width_avg);

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

        std::string line;
        int image_count = 0;
        std::string default_word;
        std::vector<std::string> words_candidates = {""};

        for (int i = 0; i < (int)components.size(); ++i) {
            if (m_debug) {
                components[i].second.save(m_debug_output_tmp / std::to_string(line_num) / (std::to_string(image_count) + ".png"));
                ++image_count;
            }

            update_candidates_and_default_word(words_candidates, default_word, components[i].second, height_avg);

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

    std::optional<image> bfs_segment_line(const image& img, pixel p) const
    {
        // create new image that contains only one line using flood fill algorithm
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

        // it is possible that we have only white pixels between separating line paths
        // we should skip that line
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

    void get_lines_text(const image& img) const
    {
        std::ofstream output(m_output_file_tmp);

        int line_num = 0;
        for (int j = 1; j < img.rows(); ++j) {
            // gray color represents separating line path
            if (j == 1 || img.check_color({j-1,0}, Color::gray)) {
                auto maybe_line = bfs_segment_line(img, {j,0});

                // skip if line contains only white pixels
                if (maybe_line) {
                    if (m_debug) {
                        maybe_line->save(m_debug_output_tmp / ("line" + std::to_string(line_num) + ".png"));
                    }
                    auto line_text = get_line_text(*maybe_line, line_num);

                    if (m_use_spell_check) {
                        // pass the entire line text to spell checker
                        auto line_fixed = m_corrector.FixFragment(std::wstring(line_text.begin(), line_text.end()));
                        std::cout << std::string(line_fixed.begin(), line_fixed.end()) << std::endl;
                        output << std::string(line_fixed.begin(), line_fixed.end()) << std::endl;
                    }
                    else {
                        std::cout << line_text << std::endl;
                        output << line_text << std::endl;
                    }

                    ++line_num;
                }
            }
        }
    }

    std::pair<std::string, std::string> recognize(image& component) const
    {
        // if model classifies component as one letter, return only one letter prediction
        // if model classifies component as two letters, return one letter prediction and two letters prediction
        component.resize(28,28);
        auto t = component.get_tensor(0,1);
        // one letter prediction
        std::string character =  std::string(1, char(m_models.model_one.predict_class({t}) + 97));

        if (m_models.model_one_two.predict_class({t}) == 1) {
            // two letters case
            return  // one letter
                    {character, 
                    // two letters
                    std::string(1, char(m_models.model_first.predict_class({t}) + 97)) + std::string(1, char(m_models.model_second.predict_class({t}) + 97))};
        }

        return {character, ""};
    }

private:
    // models for letters recognition
    recognition_models m_models;
    // spell checker model
    NJamSpell::TSpellCorrector m_corrector;
    // dictionary of english words
    std::unordered_set<std::string> m_dictionary;
    // indicates if spell checker should be used
    bool m_use_spell_check;
    // indicates if intermediate results should be saved
    bool m_debug;
    // output directory
    fs::path m_output_dir;

    // debug output directory for the passed image
    fs::path m_debug_output_tmp;
    // output file for the passed image
    fs::path m_output_file_tmp;
};


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "./keras2cpp path_to_img [dictionary] [debug]" << std::endl;
        return -1;
    }

    recognition_models models;


    if (fs::is_directory(argv[1])) {
        image_recognizer recognizer_dictionary;
        image_recognizer recognizer(false);

        for (auto it = fs::directory_iterator(argv[1]); it != fs::directory_iterator(); ++it) {
            if (fs::path(*it).extension() == ".png") {
                recognizer.recognize_image(*it);
                recognizer_dictionary.recognize_image(*it);
            }
        }
    }
    else {
        bool use_dictionary = false;
        bool debug = false;

        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "dictionary") {
                use_dictionary = true;
            }
            else if (std::string(argv[i]) == "debug") {
                debug = true;
            }
        }

        image_recognizer recognizer(use_dictionary, debug);
        recognizer.recognize_image(argv[1]);
    }

    return 0;
}
