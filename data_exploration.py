import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa.display


def get_sounds(data, labels, cats):
    sounds = []

    for t in range(5):
        sounds.append(np.count_nonzero(labels == t))
        print(f"There are {sounds[t]} spectrograms of {cats[t]}")

    sounds.append(len(labels))

    all_cats = np.zeros((32, 96))
    other = np.zeros((32, 96))
    music = np.zeros((32, 96))
    human_voice = np.zeros((32, 96))
    engine_sounds = np.zeros((32, 96))
    alarm = np.zeros((32, 96))
    # Find means
    for i in range(len(data)):
        all_cats += data[i]
        if labels[i] == 0:
            other += data[i]
        elif labels[i] == 1:
            music += data[i]
        elif labels[i] == 2:
            human_voice += data[i]
        elif labels[i] == 3:
            engine_sounds += data[i]
        elif labels[i] == 4:
            alarm += data[i]

    return sounds, all_cats, other, music, human_voice, engine_sounds, alarm


def plot_means(data, labels, cats, save_plot=False):

    sounds, all_cats, other, music, human_voice, engine_sounds, alarm = get_sounds(data, labels, cats)

    # mean
    all_mean = all_cats / sum(sounds)
    other_mean = other / sounds[0]
    music_mean = music / sounds[1]
    human_voice_mean = human_voice / sounds[2]
    engine_sounds_mean = engine_sounds / sounds[3]
    alarm_mean = alarm / sounds[4]

    means = [other_mean, music_mean, human_voice_mean, engine_sounds_mean, alarm_mean, all_mean]

    nrows, ncols = 2, 3
    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.1])
    gs.update(wspace=0.2, hspace=0.4)

    axes = []
    mean_nr = 0
    for c in range(nrows):
        for r in range(ncols):
            mel_spectrogram = means[mean_nr]
            # Plotting the mel-spectrogram
            ax = plt.subplot(gs[c, r])
            ax.set_xlim(0, 2)
            img = librosa.display.specshow(mel_spectrogram, y_axis='mel', x_axis='time', cmap='viridis', ax=ax)

            if c+r == 3:
                ax.set_title(f'Mean mel-spectrogram of all the 52890 spectrograms')
            else:
                ax.set_title(f'Mean mel-spectrogram of all the {sounds[mean_nr]} {cats[mean_nr]} spectrograms ')
            axes.append(ax)

            # Add row title
            row_title_ax = fig.add_subplot(gs[c, 0], frameon=False, label=f'row_title_ax_{c}_{r}')
            row_title_ax.set_xticks([])
            row_title_ax.set_yticks([])

            # Remove y-axis tick labels for all but the left-most plots
            if mean_nr != 0 and mean_nr != 3:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('Frequency [Hz]')

            # Remove x-axis tick labels for all but the bottom plots
            if mean_nr < 3:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time [s]')

            mean_nr += 1

    # Adding a colorbar
    cbar_ax = plt.subplot(gs[:, -1])
    fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')

    # Adding the overall title
    # fig.suptitle("The mean spectrogram of each of the categories", fontsize=24)
    plt.subplots_adjust(top=0.92)
    if save_plot:
        plt.savefig("mean_mel_spectrograms.pdf", bbox_inches="tight")
    plt.show()


def get_5x5(data, labels, cats, save_plot=False):
    nrows, ncols = 5, 5
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.1])
    gs.update(wspace=0.2, hspace=0.4)

    axes = []
    for c in range(nrows):
        i = 0
        counter = 0
        while counter < ncols:
            if labels[i] == c:
                mel_spectrogram = data[i]
                # Plotting the mel-spectrogram
                ax = plt.subplot(gs[c, counter])
                ax.set_xlim(0, 2)

                # img = ax.imshow(np.flip(mel_spectrogram, axis=0), interpolation="nearest")
                img = librosa.display.specshow(mel_spectrogram, y_axis='mel', x_axis='time', cmap='viridis', ax=ax)

                ax.set_title(f'Spectrogram {i}', fontsize=10)
                counter += 1

                # Remove y-axis tick labels for all but the left-most plots
                if counter > 1:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel('Frequency [Hz]')

                # Remove x-axis tick labels for all but the bottom plots
                if c < nrows - 1:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Time [s]')

            i = i + 1
        axes.append(ax)

        # Add row title
        row_title_ax = fig.add_subplot(gs[c, 0], frameon=False)
        row_title_ax.set_xticks([])
        row_title_ax.set_yticks([])
        row_title_ax.text(-0.4, 0.5, cats[c], fontsize=12, ha='right', va='center', rotation='horizontal',
                          transform=row_title_ax.transAxes)

    # Adding a colorbar
    cbar_ax = plt.subplot(gs[:, -1])
    fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')

    # Adding the overall title
    fig.suptitle("5 Mel-Spectrograms for each of the 5 sound categories", fontsize=20)
    plt.subplots_adjust(top=0.92)
    if save_plot:
        plt.savefig("mel_spectrograms.pdf", bbox_inches="tight")
    plt.show()


def get_3x5(data, labels, cats, save_plot=False):
    nrows, ncols = 3, 5
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.1])
    gs.update(wspace=0.2, hspace=0.2)

    axes = []

    the_specs = [7, 0, 3, 22, 66, 16, 2, 46, 26, 166, 17, 5, 61, 39, 168]
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            mel_spectrogram = data[the_specs[i]]
            # Plotting the mel-spectrogram
            ax = plt.subplot(gs[r, c])
            ax.set_xlim(0, 2)

            img = librosa.display.specshow(mel_spectrogram, y_axis='mel', x_axis='time', cmap='viridis', ax=ax)

            # Remove y-axis tick labels for all but the left-most plots
            if c > 0:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('Frequency [Hz]')

            # Remove x-axis tick labels for all but the bottom plots
            if r < nrows - 1:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time [s]')

            # Add column titles for the first row
            if r == 0:
                ax.set_title(cats[c], fontsize=12)

            i = i + 1
        axes.append(ax)

    # Adding a colorbar
    cbar_ax = plt.subplot(gs[:, -1])
    fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')

    plt.subplots_adjust(top=0.92)
    if save_plot:
        plt.savefig("mel_spectrograms_3_5.pdf", bbox_inches="tight")
    plt.show()


def plot_distributions(data, labels, cats):

    # Flatten the data array
    flattened_data = data.ravel()

    plt.style.use('seaborn-darkgrid')

    # Create the histogram
    plt.hist(flattened_data, bins=np.arange(-80, 1, 1), alpha=0.7, edgecolor='black', linewidth=0.5, color='steelblue')


    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig("mel_spectrograms_distribution.pdf", bbox_inches="tight")
    plt.title('Distribution of values in the spectrograms')

    # Show the plot
    plt.show()

    other = []
    music = []
    human_voice = []
    engine_sounds = []
    alarm = []

    for i, label in enumerate(labels):
        if label == 0:
            other.append(data[i])
        elif label == 1:
            music.append(data[i])
        elif label == 2:
            human_voice.append(data[i])
        elif label == 3:
            engine_sounds.append(data[i])
        elif label == 4:
            alarm.append(data[i])

    class_data = [np.array(other), np.array(music), np.array(human_voice), np.array(engine_sounds), np.array(alarm)]

    for i in range(len(class_data)):
        # Flatten the data array
        flattened_data = class_data[i].ravel()


        # Create the histogram
        plt.hist(flattened_data, bins=np.arange(-80, 1, 1), alpha=0.7, edgecolor='black')

        # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {cats[i]} values')

        # Show the plot
        plt.show()


def count_80s(data, labels, cats):

    data_r = data.round()

    other_80s = []
    music_80s = []
    human_voice_80s = []
    engine_sounds_80s = []
    alarm_80s = []

    class_80s = [other_80s, music_80s, human_voice_80s, engine_sounds_80s, alarm_80s]

    for i in range(len(data_r)):
        count = np.count_nonzero(data_r[i] == -80)
        if count != 0:
            class_80s[labels[i]].append(count)

    for i in range(len(class_80s)):

        # Create the histogram
        plt.hist(class_80s[i], bins=100, alpha=0.7, edgecolor='black')

        # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {cats[i]} values')

        # Show the plot
        plt.show()

    # Set the number of bins for the histograms
    num_bins = 50

    # Create subplots with 2 rows and 3 columns
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True, sharey=True)

    # Flatten the axes array for easier indexing
    flat_axes = axes.flatten()

    # Create the histograms
    flat_axes[0].hist(class_80s[0], bins=num_bins)
    flat_axes[0].set_title('Data 1')

    flat_axes[1].hist(class_80s[1], bins=num_bins)
    flat_axes[1].set_title('Data 2')

    flat_axes[2].hist(class_80s[2], bins=num_bins)
    flat_axes[2].set_title('Data 3')

    flat_axes[3].hist(class_80s[3], bins=num_bins)
    flat_axes[3].set_title('Data 4')

    flat_axes[4].hist(class_80s[4], bins=num_bins)
    flat_axes[4].set_title('Data 5')

    # Remove the last (empty) subplot
    fig.delaxes(flat_axes[5])

    # Set global labels and title
    fig.text(0.5, 0.04, 'Value', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    fig.suptitle('Histograms of 5 Datasets')

    # Adjust the layout for better appearance
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Display the plot
    plt.show()

if __name__ == '__main__':
    D = np.load("training.npy")
    L = np.load("training_labels.npy")
    C = ["Other", "Music", "Human voice", "Engine sounds", "Alarm", "All"]

    plot_means(D, L, C, save_plot=False)
    get_5x5(D, L, C, save_plot=False)
    get_3x5(D, L, C, save_plot=False)
    plot_distributions(D, L, C)
    count_80s(D, L, C)
