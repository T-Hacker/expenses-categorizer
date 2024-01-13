use neuroflow::{data::DataSet, FeedForward};

#[derive(Debug)]
struct Entry {
    date_mov: String,
    date_val: String,
    description: String,
    debit: String,
    credit: String,
    category: String,
}

fn main() {
    let entries = read_data("example_data.tsv");
    dbg!(&entries);

    // Input format:
    // date_mov:    5
    // date_val:    5
    // description: 21
    // debit:       10
    // credit:      10
    // TOTAL:       51

    let num_training_entries = (entries.len() as f32 * 0.8).floor() as usize;
    let num_testing_entries = entries.len() - num_training_entries;

    let mut nn = FeedForward::new(&[51, 70, 100, 70, 40, 1]);

    let mut data = DataSet::new();
    for entry in entries.iter().take(num_training_entries) {}
}

fn read_data(file: &str) -> Vec<Entry> {
    let mut entries = vec![];

    let data = std::fs::read_to_string(file).unwrap();
    let data = data.lines().skip(1);
    for line in data {
        let mut columns = line.split('\t');

        entries.push(Entry {
            date_mov: columns.next().unwrap().into(),
            date_val: columns.next().unwrap().into(),
            description: columns.next().unwrap().into(),
            debit: columns.next().unwrap().into(),
            credit: columns.next().unwrap().into(),
            category: columns.skip(2).next().unwrap().into(),
        });
    }

    entries
}
