use neuroflow::{
    activators::Type::{self},
    data::{DataSet, Extractable},
    FeedForward,
};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::{collections::HashSet, time::Instant};

#[derive(Debug)]
struct Entry {
    date_mov: String,
    date_val: String,
    description: String,
    debit: String,
    credit: String,
    category: String,
}

// Input format:
// date_mov:    5
// date_val:    5
// description: 21
// debit:       10
// credit:      10
// TOTAL:       51

fn main() {
    // Process the input data.
    let mut entries = read_data("example_data.csv");
    entries.shuffle(&mut thread_rng());
    // dbg!(&entries);

    // Generate categories from the entries.
    let categories: HashSet<_> = entries
        .iter()
        .map(|entry| entry.category.as_str())
        .collect();
    let mut categories: Vec<_> = categories.into_iter().collect();
    categories.sort();
    let categories: &[&str] = categories.as_slice();

    // Split the data into training and testing sets.
    let num_training_entries = (entries.len() as f32 * 0.8).floor() as usize;
    let num_testing_entries = entries.len() - num_training_entries;

    let mut data = DataSet::new();
    for entry in entries.iter().take(num_training_entries) {
        convert_entry_to_dataset(entry, categories, &mut data);
    }

    // Training model.
    println!("Training model...");
    let (input, label) = data.get(0);
    let input_size = input.len() as i32;
    let label_size = label.len() as i32;
    println!("Input size: {input_size}, Label size: {label_size}");

    let mut nn = FeedForward::new(&[input_size, 70, 100, 70, 40, categories.len() as i32]);
    let now = Instant::now();
    nn.activation(Type::Sigmoid)
        .learning_rate(0.01)
        .train(&data, 1_000_000);

    println!(
        "Took {} seconds to train with error of {}.",
        now.elapsed().as_secs_f32(),
        nn.get_error()
    );

    // // Run tests.
    // println!("Running tests...");
    // let mut diff_avg = 0.0;
    // for entry in entries
    //     .iter()
    //     .skip(num_training_entries)
    //     .take(num_testing_entries)
    // {
    //     let input = normalize_input(&entry);
    //     let res = nn.calc(&input);
    //     dbg!(res);
    //
    //     let categories_probability = normalize_category(&entry.category, categories);
    //     dbg!(categories_probability);
    //
    //     // let diff = res - category_index;
    //     //
    //     // diff_avg += diff;
    //     //
    //     // let category_index = f64::round(category_index) as usize;
    //     // match categories.get(category_index) {
    //     //     Some(category) => println!("Predicted: {} was {}", category, entry.category),
    //     //     None => println!(
    //     //         "Invalid category: {} was {}",
    //     //         category_index, entry.category
    //     //     ),
    //     // }
    // }
    //
    // diff_avg /= num_testing_entries as f64;
    // println!("Average diff: {diff_avg}");
}

fn read_data(file: &str) -> Vec<Entry> {
    let mut entries = vec![];

    let mut rdr = csv::Reader::from_path(file).unwrap();
    for result in rdr.records() {
        let record = result.unwrap();
        let mut record = record.iter();

        entries.push(Entry {
            date_mov: record.next().unwrap().into(),
            date_val: record.next().unwrap().into(),
            description: record.next().unwrap().into(),
            debit: record.next().unwrap().into(),
            credit: record.next().unwrap().into(),
            category: record.skip(2).next().unwrap().into(),
        });
    }

    entries
}

fn convert_entry_to_dataset(entry: &Entry, categories: &[&str], dataset: &mut DataSet) {
    let input = normalize_input(entry);
    assert_eq!(input.len(), 29);

    let labels = normalize_category(&entry.category, categories);

    dataset.push(&input, &labels);
}

fn normalize_input(entry: &Entry) -> Vec<f64> {
    let mut input = Vec::with_capacity(51);

    normalize_date(&entry.date_mov, &mut input);
    normalize_date(&entry.date_val, &mut input);
    normalize_string::<21>(&entry.description, &mut input);
    normalize_currency(entry.debit.clone(), &mut input);
    normalize_currency(entry.credit.clone(), &mut input);

    input
}

fn normalize_date(date: &str, values: &mut Vec<f64>) {
    let mut date_elements = date.split('-');

    let day = date_elements.next().unwrap().parse::<f64>().unwrap();
    let day = normalize(day, 1.0, 31.0);
    values.push(day);

    let month = date_elements.next().unwrap().parse::<f64>().unwrap();
    let month = normalize(month, 1.0, 12.0);
    values.push(month);

    let year = date_elements.next().unwrap().parse::<f64>().unwrap();
    let year = normalize(year, 1970.0, 2040.0);
    values.push(year);
}

fn normalize_string<const SIZE: i32>(text: &str, values: &mut Vec<f64>) {
    let mut count = 0;
    for mut c in text.chars() {
        c.make_ascii_uppercase();

        let value = c as u32;
        let value = value as f64;
        let value = normalize(value, 48.0, 90.0);

        values.push(value);

        count += 1;
    }

    let mut num_chars_to_fill = SIZE - count;
    assert!(num_chars_to_fill >= 0);

    while num_chars_to_fill > 0 {
        values.push(-1.0);

        num_chars_to_fill -= 1;
    }
}

fn normalize_currency(mut text: String, values: &mut Vec<f64>) {
    if text.is_empty() {
        values.push(-1.0);

        return;
    }

    text.retain(|c| c != '.');
    text = text.replace(",", ".");

    values.push(text.parse().unwrap());
}

fn normalize_category(text: &str, categories: &[&str]) -> Vec<f64> {
    let mut values = Vec::with_capacity(categories.len());

    let index = categories.iter().position(|c| *c == text).unwrap();
    for i in 0..categories.len() {
        values.push(if i == index { 1.0 } else { 0.0 });
    }

    values
}

fn normalize(value: f64, min: f64, max: f64) -> f64 {
    (value - min) / (max - min)
}
