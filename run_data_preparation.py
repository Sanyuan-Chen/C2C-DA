# coding:utf-8

""" Cluster-to-Cluster Pairs Construction script"""

import argparse
import os
import json
import random
import edit_distance


def k_mediods(dataset, k, edit_distance_matrix, sent2idx_dict):

    def assign_points(data_points, centers, edit_distance_matrix, sent2idx_dict):
        assignments = []
        for point in data_points:
            shortest = 999999  # positive infinity
            shortest_index = 0
            for i, center in enumerate(centers):
                idx_p = sent2idx_dict[' '.join(point)]
                idx_c = sent2idx_dict[' '.join(center)]
                if edit_distance_matrix[idx_p][idx_c] >= 0:
                    dist = edit_distance_matrix[idx_p][idx_c]
                else:
                    sm = edit_distance.SequenceMatcher(a=point, b=center)
                    dist = sm.distance()
                    edit_distance_matrix[idx_p][idx_c] = dist
                    edit_distance_matrix[idx_c][idx_p] = dist
                if dist < shortest:
                    shortest = dist
                    shortest_index = i
            assignments.append(shortest_index)
        return assignments

    def update_centers(data_set, assignments, edit_distance_matrix, sent2idx_dict):
        new_means = {}
        centers = []
        for assignment, point in zip(assignments, data_set):
            if assignment not in new_means:
                new_means[assignment] = [point]
            else:
                new_means[assignment].append(point)

        for center in new_means:
            points = new_means[center]
            shortest = 999999  # positive infinity
            shortest_p = []
            for i, point in enumerate(points):
                total_dist = 0
                for j, point2 in enumerate(points):
                    idx_p = sent2idx_dict[' '.join(point)]
                    idx_p2 = sent2idx_dict[' '.join(point2)]
                    if edit_distance_matrix[idx_p][idx_p2] >= 0:
                        dist = edit_distance_matrix[idx_p][idx_p2]
                    else:
                        sm = edit_distance.SequenceMatcher(a=point, b=point2)
                        dist = sm.distance()
                        edit_distance_matrix[idx_p][idx_p2] = dist
                        edit_distance_matrix[idx_p2][idx_p] = dist
                    total_dist += dist
                if total_dist < shortest:
                    shortest = total_dist
                    shortest_p = point
            centers.append(shortest_p)
        return centers

    k_points = random.sample(dataset, k)
    assignments = assign_points(dataset, k_points, edit_distance_matrix, sent2idx_dict)

    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments, edit_distance_matrix, sent2idx_dict)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers, edit_distance_matrix, sent2idx_dict)

    assignments = zip(assignments, dataset)
    clusters = {}
    for idx_c, data in assignments:
        if idx_c not in clusters:
            clusters[idx_c] = [data]
        else:
            clusters[idx_c].append(data)
    return assignments, clusters


def target_cluster_sampling(args, source_clusters, edit_distance_matrix, sent2idx_dict):

    def target_sentence_sampling(s_cluster, candidates, edit_distance_matrix, sent2idx_dict):
        longest = -1
        longest_p = []
        longest_p_nearest_p = []
        for point in candidates:
            idx_p = sent2idx_dict[' '.join(point)]
            min_dist = 999999
            min_p = []
            for s_point in s_cluster:
                idx_sp = sent2idx_dict[' '.join(s_point)]
                if edit_distance_matrix[idx_p][idx_sp] >= 0:
                    dist = edit_distance_matrix[idx_p][idx_sp]
                else:
                    sm = edit_distance.SequenceMatcher(a=point, b=s_point)
                    dist = sm.distance()
                    edit_distance_matrix[idx_p][idx_sp] = dist
                    edit_distance_matrix[idx_sp][idx_p] = dist
                if dist < min_dist:
                    min_p = s_point
                min_dist = min(dist, min_dist)
            if min_dist > longest:
                longest = min_dist
                longest_p = point
                longest_p_nearest_p = min_p
        return longest_p, longest, longest_p_nearest_p

    target_clusters = {}
    for idx_c in source_clusters:
        s_cluster = source_clusters[idx_c]
        t_cluster = []
        while len(t_cluster) < args.target_cluster_num_threshold:
            candidates = []
            for idx_c2 in source_clusters:
                if idx_c != idx_c2:
                    candidates += source_clusters[idx_c2]
            target_sentence, target_dist, nearest_sentence = target_sentence_sampling(s_cluster + t_cluster, candidates, edit_distance_matrix, sent2idx_dict)
            if target_dist < args.target_cluster_dist_threshold and len(t_cluster) > 0:
                break
            t_cluster.append(target_sentence)
        target_clusters[idx_c] = t_cluster

    return target_clusters


def data_load_and_classify(args, file_name, file_path):
    # load and classify the data by slots value
    print('Loading and classifying the data by slots value......')
    data_classified = {}
    slot2entity_dict = {}

    sentence = []
    sentence_slots = set()
    idx_debug = 0
    raw_file = open(file_path, 'r')
    for raw_data_line in raw_file:
        if not raw_data_line.isspace():
            line = raw_data_line.split()
            if line[1][0] == 'O':
                sentence.append(line[0])
            elif line[1][0] == 'B':
                slot = '<' + line[1][2:] + '>'
                sentence.append(slot)
                sentence_slots.add(slot)
                if slot not in slot2entity_dict:
                    slot2entity_dict[slot] = [line[0]]
                else:
                    slot2entity_dict[slot].append(line[0])
            elif line[1][0] == 'I':
                slot = '<' + line[1][2:] + '>'
                slot2entity_dict[slot][-1] = ' '.join([slot2entity_dict[slot][-1], line[0]])
        else:
            slots_key = ' & '.join(sorted(list(sentence_slots)))
            if not slots_key:
                continue
            elif slots_key not in data_classified:
                data_classified[slots_key] = [sentence]
            elif sentence not in data_classified[slots_key]:
                data_classified[slots_key].append(sentence)

            if args.debug_mode and idx_debug < 5:
                print("line ", idx_debug + 1)
                print("sentence: ", sentence)
                print("sentence slots: ", sentence_slots)
                print("data_classified", data_classified)
                print("slot_dict: ", slot2entity_dict)
                print('\n')
                idx_debug += 1

            sentence = []
            sentence_slots = set()

    if sentence:
        slots_key = ' & '.join(sorted(list(sentence_slots)))
        if not slots_key:
            pass
        elif slots_key not in data_classified:
            data_classified[slots_key] = [sentence]
        elif sentence not in data_classified[slots_key]:
            data_classified[slots_key].append(sentence)

    if args.debug_mode:
        num_data_classified = [len(data_classified[c]) for c in data_classified]
        print("Number of unique sentences for each class: {}, {} of which greater than {}, Max number: {}, Len: {}, \n"
              .format(num_data_classified,
                      sum([1 if n > (2 * args.num_per_source_cluster - 1) else 0 for n in num_data_classified]),
                      (2 * args.num_per_source_cluster - 1), max(num_data_classified), len(num_data_classified)))

    if not os.path.exists(args.output_cluster_dir):
        os.mkdir(args.output_cluster_dir)
    with open(os.path.join(args.output_cluster_dir, file_name + '_classified_data.json'), 'w') as data_classified_file:
        json.dump(data_classified, data_classified_file, indent=4)
    for slot in slot2entity_dict:
        slot2entity_dict[slot] = list(set(slot2entity_dict[slot]))
    with open(os.path.join(args.output_cluster_dir, file_name + '_slot2entity_dictionary.json'),
              'w') as slot2entity_dict_file:
        json.dump(slot2entity_dict, slot2entity_dict_file, indent=4)
    with open(os.path.join(args.output_cluster_dir, file_name + '_all_sentences.txt'), 'w') as all_sentences_file:
        for slot in sorted(data_classified.keys()):
            for sent in data_classified[slot]:
                all_sentences_file.write(' '.join(sent) + '\n')
    print('Classified the data by slots value.\n')

    return data_classified, slot2entity_dict


def data_cluster(args, file_name, data_classified):
    # clustering the data by edit-distance for each class
    print('Clustering the data by edit-distance for each class......')
    data_clustered_classified = {}
    idx_debug = 0
    for class_name in sorted(data_classified.keys()):
        sentences = data_classified[class_name]
        n = len(sentences)
        if n < args.num_per_source_cluster * 2:
            continue

        sent2idx_dict = {}
        idx2sent_dict = {}
        for sent in sentences:
            idx2sent_dict[len(sent2idx_dict)] = sent
            sent2idx_dict[' '.join(sent)] = len(sent2idx_dict)
        assert len(sent2idx_dict) == n and len(idx2sent_dict) == n
        edit_distance_matrix = [[-1 for __ in range(n)] for _ in range(n)]
        for i in range(n):
            edit_distance_matrix[i][i] = 0

        sentences_assignments, source_clusters = k_mediods(sentences, n // args.num_per_source_cluster,
                                                           edit_distance_matrix, sent2idx_dict)

        target_clusters = target_cluster_sampling(args, source_clusters, edit_distance_matrix, sent2idx_dict)

        assert len(source_clusters) == len(target_clusters)
        for idx_c in source_clusters:
            if len(source_clusters[idx_c]) == 0 or len(target_clusters[idx_c]) == 0:
                print('warning empty:', source_clusters, target_clusters)
                exit(0)
        data_clustered_classified[class_name] = {'source_clusters': source_clusters, 'target_clusters': target_clusters}

        if args.debug_mode and idx_debug < 5:
            print('cluster ', idx_debug + 1)
            print('class_name: ', class_name)
            print('edit_distance_matrix: ', edit_distance_matrix)
            print('source_clusters:', source_clusters)
            print('target_clusters: ', target_clusters)
            print('\n')
            idx_debug += 1

    if args.debug_mode:
        num_data_clustered = [(len(data_clustered_classified[cls]['source_clusters'][cluster]),
                               len(data_clustered_classified[cls]['target_clusters'][cluster]))
                              for cls in data_clustered_classified for cluster in
                              data_clustered_classified[cls]['source_clusters']]
        print("Number of unique sentences for each cluster: {}, Max number: {}, Len: {}, \n"
              .format(num_data_clustered, max(num_data_clustered), len(num_data_clustered)))

    with open(os.path.join(args.output_cluster_dir, file_name + '_classified_clustered_data.json'),
              'w') as data_clustered_classified_file:
        json.dump(data_clustered_classified, data_clustered_classified_file, indent=4)
    print('Clustered the data by edit-distance for each class.\n')

    return data_clustered_classified


def data_split(args, file_name, data_clustered_classified):
    # split training data and generating cluster-to-cluster pairs
    print('Split training data and generating cluster-to-cluster data......')
    if not os.path.exists(args.output_pairs_dir):
        os.mkdir(args.output_pairs_dir)

    pair_list = []
    for class_name in sorted(data_clustered_classified.keys()):
        for cluster in data_clustered_classified[class_name]['source_clusters']:
            pair_list.append((class_name, cluster))
    pair_list = random.sample(pair_list, len(pair_list))
    len_pair_list = len(pair_list)
    print(pair_list, len_pair_list)

    if args.cross_generation:
        cross_list = [str(i) for i in range(args.cross_generation_num)]
    else:
        cross_list = ['']

    for idx_cross in cross_list:
        data_clustered_classified_train_src_txt = open(
            os.path.join(args.output_pairs_dir, file_name + '_clustered_train{}_src.txt'.format(idx_cross)), 'w')
        data_clustered_classified_train_tgt_txt = open(
            os.path.join(args.output_pairs_dir, file_name + '_clustered_train{}_tgt.txt'.format(idx_cross)), 'w')
        data_clustered_classified_reserve_src_txt = open(
            os.path.join(args.output_pairs_dir, file_name + '_clustered_reserve{}_src.txt'.format(idx_cross)), 'w')
        data_clustered_classified_reserve_tgt_txt = open(
            os.path.join(args.output_pairs_dir, file_name + '_clustered_reserve{}_tgt.txt'.format(idx_cross)), 'w')

        data_clustered_classified_train_src = {}
        data_clustered_classified_train_tgt = {}
        data_clustered_classified_reserve_src = {}
        data_clustered_classified_reserve_tgt = {}
        for class_name in sorted(data_clustered_classified.keys()):
            data_clustered_classified_train_src[class_name] = {}
            data_clustered_classified_train_tgt[class_name] = {}
            data_clustered_classified_reserve_src[class_name] = {}
            data_clustered_classified_reserve_tgt[class_name] = {}

        idx_split = int(idx_cross) if idx_cross != '' else 0
        train_list = pair_list[:len_pair_list // args.cross_generation_num * idx_split] + \
                     pair_list[len_pair_list // args.cross_generation_num * (idx_split + 1):
                               len_pair_list // args.cross_generation_num *
                               min((idx_split + 1 + args.cross_generation_num - 1), args.cross_generation_num)]
        reserve_list = pair_list[len_pair_list // args.cross_generation_num *
                                 idx_split: len_pair_list // args.cross_generation_num * (idx_split + 1)]
        left_list = pair_list[len_pair_list // args.cross_generation_num * args.cross_generation_num:]
        for idx_left, left in enumerate(left_list):
            if idx_split == idx_left:
                reserve_list.append(left)
            else:
                train_list.append(left)

        print(train_list, len(train_list))
        print(reserve_list, len(reserve_list))

        for class_name, cluster in train_list:
            pair_clusters = data_clustered_classified[class_name]
            data_clustered_classified_train_src[class_name][cluster] = pair_clusters['source_clusters'][cluster]
            data_clustered_classified_train_src_txt.write(
                '\t'.join([' '.join(sent) for sent in pair_clusters['source_clusters'][cluster]]) + '\n')
            data_clustered_classified_train_tgt[class_name][cluster] = pair_clusters['target_clusters'][cluster]
            data_clustered_classified_train_tgt_txt.write(
                '\t'.join([' '.join(sent) for sent in pair_clusters['target_clusters'][cluster]]) + '\n')
        for class_name, cluster in reserve_list:
            pair_clusters = data_clustered_classified[class_name]
            data_clustered_classified_reserve_src[class_name][cluster] = pair_clusters['source_clusters'][cluster]
            data_clustered_classified_reserve_src_txt.write(
                '\t'.join([' '.join(sent) for sent in pair_clusters['source_clusters'][cluster]]) + '\n')
            data_clustered_classified_reserve_tgt[class_name][cluster] = pair_clusters['target_clusters'][cluster]
            data_clustered_classified_reserve_tgt_txt.write(
                '\t'.join([' '.join(sent) for sent in pair_clusters['target_clusters'][cluster]]) + '\n')

        with open(os.path.join(args.output_pairs_dir, file_name + '_clustered_train{}_src.json'.format(idx_cross)),
                  'w') as data_clustered_classified_train_src_file:
            json.dump(data_clustered_classified_train_src, data_clustered_classified_train_src_file, indent=4)
        with open(os.path.join(args.output_pairs_dir, file_name + '_clustered_train{}_tgt.json'.format(idx_cross)),
                  'w') as data_clustered_classified_train_tgt_file:
            json.dump(data_clustered_classified_train_tgt, data_clustered_classified_train_tgt_file, indent=4)
        with open(os.path.join(args.output_pairs_dir, file_name + '_clustered_reserve{}_src.json'.format(idx_cross)),
                  'w') as data_clustered_classified_reserve_src_file:
            json.dump(data_clustered_classified_reserve_src, data_clustered_classified_reserve_src_file, indent=4)
        with open(os.path.join(args.output_pairs_dir, file_name + '_clustered_reserve{}_tgt.json'.format(idx_cross)),
                  'w') as data_clustered_classified_reserve_tgt_file:
            json.dump(data_clustered_classified_reserve_tgt, data_clustered_classified_reserve_tgt_file, indent=4)
        data_clustered_classified_train_src_txt.close()
        data_clustered_classified_train_tgt_txt.close()
        data_clustered_classified_reserve_src_txt.close()
        data_clustered_classified_reserve_tgt_txt.close()

    print('Split completed.\n')


def file_clustering(args, file_name, file_path):
    print('Clustering the file {}......\n'.format(file_name))

    data_classified, slot2entity_dict = data_load_and_classify(args, file_name, file_path)

    data_clustered_classified = data_cluster(args, file_name, data_classified)

    data_split(args, file_name, data_clustered_classified)


def run_clustering(args):
    file_name = os.path.split(args.data_path)[-1]
    file_clustering(args, file_name, args.data_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data_path", type=str, default='./data/Atis/atis_train',
                        help="path of the original training data")
    parser.add_argument("-ocd", "--output_cluster_dir", type=str, default='./data/Cluster',
                        help="output directory of the cluster files")
    parser.add_argument("-opd", "--output_pairs_dir", type=str, default='./data/AugmentedData',
                        help="output directory of the cluster-to-cluster paraphrasing pairs")

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("-npc", "--num_per_source_cluster", type=int, default=5,
                        help="Avg. number per source cluster")
    parser.add_argument("-tcdt", "--target_cluster_dist_threshold", type=int, default=4,
                        help="distance threshold of target cluster selection")
    parser.add_argument("-tcnt", "--target_cluster_num_threshold", type=int, default=10,
                        help="quantity threshold of target cluster selection")
    parser.add_argument("-cg", "--cross_generation", action='store_true', help="Whether to use cross generation.")
    parser.add_argument("-cgn", "--cross_generation_num", type=int, default=5)

    parser.add_argument("--debug_mode", action='store_true', help="Whether to run debug mode.")

    args = parser.parse_args()

    random.seed(args.seed)

    run_clustering(args)


if __name__ == "__main__":
    main()
