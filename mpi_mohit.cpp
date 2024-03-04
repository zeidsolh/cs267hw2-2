#include "common.h"
#include <cmath>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <functional>

using std::vector;

//Compute forces using bidirectional property
void compute_force(particle_t& particle, particle_t& neighbor, bool bidirectional) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
    if (bidirectional) {
        neighbor.ax -= coef * dx;
        neighbor.ay -= coef * dy;
    }
}

void update_position(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

class ParticleGroup {
  public:
    int group_id, row, col, rows, cols;
    int neighbors[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    vector<vector<particle_t>> bins;
    ParticleGroup() {};
    ParticleGroup(int id);
    vector<particle_t>& get_bin(int row, int col) {
        return bins[row + 1 + (col + 1) * (rows + 2)];
    }
    void iterate_inner(std::function<void(vector<particle_t>&, int)> func) {
        for (int d : {0, 2, 4, 6}) {
            int bound = rows, other = cols;
            if (d % 4) std::swap(bound, other);
            int col = d < 4 ? 0 : other - 1;
            for (int row = 0; row < bound; row++) {
                func(d % 4 ? get_bin(col, row) : get_bin(row, col), d);
            }
            int row = d % 6 ? bound - 1 : 0;
            func(d % 4 ? get_bin(col, row) : get_bin(row, col), d + 1);
        }
    }
    void iterate_outer(std::function<void(vector<particle_t>&, int)> func) {
        for (int d : {0, 2, 4, 6}) {
            int bound = rows, other = cols;
            if (d % 4) std::swap(bound, other);
            int col = d < 4 ? -1 : other;
            for (int row = 0; row < bound; row++) {
                func(d % 4 ? get_bin(col, row) : get_bin(row, col), d);
            }
            int row = d % 6 ? bound : -1;
            func(d % 4 ? get_bin(col, row) : get_bin(row, col), d + 1);
        }
    }
    bool contains(int x, int y) {
        return 0 <= x && x < rows && 0 <= y && y < cols;
    }
};

static int grid_size, grid_rows, grid_cols;
static ParticleGroup particle_group;

void helper(int size, int count, int index, int& start, int& range) {
    int s = size / count;
    int remainder = size % count;
    start = index * s + std::min(index, remainder);
    range = s + (index < remainder ? 1 : 0);
}

ParticleGroup::ParticleGroup(int id) {
    if (id >= grid_rows * grid_cols) return;
    helper(grid_size, grid_cols, id % grid_cols, group_id, rows);
    helper(grid_size, grid_rows, id / grid_cols, col, cols);
    bool left = group_id > 0;
    bool right = group_id + rows < grid_size;
    bool up = col > 0;
    bool down = col + cols < grid_size;
    if (left) {
        neighbors[2] = id - 1;
        if (up) neighbors[1] = id - 1 - grid_cols;
        if (down) neighbors[3] = id - 1 + grid_cols;
    }
    if (right) {
        neighbors[6] = id + 1;
        if (up) neighbors[7] = id + 1 - grid_cols;
        if (down) neighbors[5] = id + 1 + grid_cols;
    }
    if (up) neighbors[0] = id - grid_cols;
    if (down) neighbors[4] = id + grid_cols;
    bins.resize((rows + 2) * (cols + 2));
}

void exchange_edges() {
    vector<MPI_Request> requests;
    requests.reserve((particle_group.rows + particle_group.cols + 2) * 2);
    auto send = [&requests](vector<particle_t>& bin, int dir) {
        if (particle_group.neighbors[dir] == -1) return;
        requests.emplace_back(MPI_Request());
        MPI_Isend(bin.data(), bin.size(), PARTICLE, particle_group.neighbors[dir], (dir + 4) % 8, MPI_COMM_WORLD,
                  &requests.back());
    };
    auto receive = [](vector<particle_t>& bin, int dir) {
        if (particle_group.neighbors[dir] == -1) return;
        MPI_Status status;
        MPI_Probe(particle_group.neighbors[dir], dir, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, PARTICLE, &count);
        bin.resize(count);
        MPI_Recv(&bin[0], count, PARTICLE, particle_group.neighbors[dir], dir, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    };
    particle_group.iterate_inner(send);
    particle_group.iterate_outer(receive);
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

void exchange_edges_2() {
    vector<MPI_Request> requests;
    requests.reserve((particle_group.rows + particle_group.cols + 2) * 2);
    auto send = [&requests](vector<particle_t>& bin, int dir) {
        if (particle_group.neighbors[dir] == -1) return;
        requests.emplace_back(MPI_Request());
        MPI_Isend(bin.data(), bin.size(), PARTICLE, particle_group.neighbors[dir], (dir + 4) % 8, MPI_COMM_WORLD,
                  &requests.back());
    };
    auto receive = [](vector<particle_t>& bin, int dir) {
        if (particle_group.neighbors[dir] == -1) return;
        MPI_Status status;
        MPI_Probe(particle_group.neighbors[dir], dir, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, PARTICLE, &count);
        auto size = bin.size();
        bin.resize(size + count);
        MPI_Recv(&bin[size], count, PARTICLE, particle_group.neighbors[dir], dir, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    };
    particle_group.iterate_outer(send);
    particle_group.iterate_inner(receive);
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

void init_simulation(particle_t* particles, int num_particles, double space_size, int rank, int num_processes) {
    grid_size = ceil(space_size / cutoff);
    grid_rows = floor(sqrt(num_processes));
    grid_cols = grid_rows + (num_processes - grid_rows * grid_rows) / grid_rows;
    if (rank >= grid_cols * grid_rows) return;
    particle_group = ParticleGroup(rank);
    for (int particle_index = 0; particle_index < num_particles; particle_index++) {
        particle_t particle = particles[particle_index];
        particle.ax = particle.ay = 0;
        int bin_row = floor(particle.x / cutoff) - particle_group.group_id;
        int bin_col = floor(particle.y / cutoff) - particle_group.col;
        if (particle_group.contains(bin_row, bin_col)) 
            particle_group.get_bin(bin_row, bin_col).push_back(particle);
    }
}

void simulate_one_step(particle_t* particles, int num_particles, double space_size, int rank, int num_processes) {
    if (rank >= grid_cols * grid_rows) return;
    // Send MPI data
    exchange_edges();
    // Compute forces across grid cells
    for (int row_index = 0; row_index < particle_group.rows; row_index++) {
        for (int col_index = 0; col_index < particle_group.cols; col_index++) {
            vector<particle_t>& bin = particle_group.get_bin(row_index, col_index);
            for (particle_t& particle : bin) {
                for (int ni : {row_index - 1, row_index, row_index + 1}) {
                    for (int nj : {col_index - 1, col_index, col_index + 1}) {
                        if (ni == row_index && nj == col_index) continue;
                        if (!particle_group.contains(ni, nj)) {
                            for (particle_t& neighbor : particle_group.get_bin(ni, nj)) {
                                compute_force(particle, neighbor, false);
                            }
                        } else if (nj > col_index || (nj == col_index && ni > row_index)) {
                            for (particle_t& neighbor : particle_group.get_bin(ni, nj)) {
                                compute_force(particle, neighbor, true);
                            }
                        }
                    }
                }
            }
            //Compute forces within grid cells
            for (int p = 0; p < bin.size(); p++) {
                for (int n = p + 1; n < bin.size(); n++) {
                    compute_force(bin[p], bin[n], true);
                }
            }
        }
    }
    struct movement {
        particle_t particle;
        int row, col;
    };
    vector<movement> intermediate_movements;
    for (int row_index = -1; row_index <= particle_group.rows; row_index++) {
        particle_group.get_bin(row_index, -1).clear();
        particle_group.get_bin(row_index, particle_group.cols).clear();
    }
    for (int col_index = -1; col_index <= particle_group.cols; col_index++) {
        particle_group.get_bin(-1, col_index).clear();
        particle_group.get_bin(particle_group.rows, col_index).clear();
    }
    // Move particles to new positions
    for (int row_index = 0; row_index < particle_group.rows; row_index++) {
        for (int col_index = 0; col_index < particle_group.cols; col_index++) {
            vector<particle_t>& bin = particle_group.get_bin(row_index, col_index);
            auto func = [row_index, col_index, space_size, &intermediate_movements](particle_t& particle){
                update_position(particle, space_size);
                particle.ax = particle.ay = 0;
                int bin_row = floor(particle.x / cutoff) - particle_group.group_id;
                int bin_col = floor(particle.y / cutoff) - particle_group.col;
                if (bin_row == row_index && bin_col == col_index) return false;
                if (particle_group.contains(bin_row, bin_col)) {
                    intermediate_movements.push_back({particle, bin_row, bin_col});
                } else {
                    particle_group.get_bin(bin_row, bin_col).push_back(particle);
                }
                return true;
            };
            bin.erase(std::remove_if(bin.begin(), bin.end(), func), bin.end());
        }
    }
    for (movement m : intermediate_movements) {
        particle_group.get_bin(m.row, m.col).push_back(m.particle);
    }
    //Send MPI data based on new positions
    exchange_edges_2();
}

void gather_for_save(particle_t* particles, int num_particles, double space_size, int rank, int num_processes) {
    if (rank >= grid_cols * grid_rows) return;
    vector<MPI_Request> requests;
    requests.reserve(particle_group.rows * particle_group.cols);
    for (int row_index = 0; row_index < particle_group.rows; row_index++) {
        for (int col_index = 0; col_index < particle_group.cols; col_index++) {
            vector<particle_t>& bin = particle_group.get_bin(row_index, col_index);
            requests.emplace_back(MPI_Request());
            MPI_Isend(bin.data(), bin.size(), PARTICLE, 0, 0, MPI_COMM_WORLD, &requests.back());
        }
    }
    if (rank == 0) {
        particle_t* particles_pointer = particles;
        for (int i = 0; i < grid_size * grid_size; i++) {
            MPI_Status status;
            MPI_Recv(particles_pointer, num_particles, PARTICLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, PARTICLE, &count);
            particles_pointer += count;
        }
        std::sort(particles, particles + num_particles, [](particle_t const &a, particle_t const &b) {
            return a.id < b.id; 
        });
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}
