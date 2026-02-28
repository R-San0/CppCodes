#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct> // 追加: Eigenのクロネッカー積
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <limits>

namespace fs = std::filesystem;

// --------------------------------------------------------------
// Eigen aliases
// --------------------------------------------------------------
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Triplet = Eigen::Triplet<double>;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

// small helpers
static inline Mat dense(const SpMat &A) { return Mat(A); }
static inline SpMat sparse(const Mat &A) { return A.sparseView(); }

// extract diagonal (as Vector)
static inline Vec diagvals_mat(const Mat &A) { return A.diagonal(); }

// write CSV (two columns)
static void save_csv_two(const std::string &path, const std::vector<double> &x,
                         const std::vector<double> &y,
                         const std::string &h1 = "omega",
                         const std::string &h2 = "value")
{
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream ofs(path);
    ofs << h1 << "," << h2 << "\n";
    ofs.setf(std::ios::fixed);
    ofs << std::setprecision(12);
    for (size_t i = 0; i < x.size(); ++i)
        ofs << x[i] << "," << y[i] << "\n";
}

// --------------------------------------------------------------
// Kronecker (sparse ⊗ sparse) -> sparse
// Eigenのサポートモジュールを使用して高速・簡略化
// --------------------------------------------------------------
static SpMat kron_sp(const SpMat &A, const SpMat &B)
{
    SpMat C = Eigen::kroneckerProduct(A, B).eval();
    C.makeCompressed();
    return C;
}

// --------------------------------------------------------------
// Local (spinful) site operators (4×4) & constants
// Basis: |0>, |↑>, |↓>, |↑↓>  (apply ↑ before ↓ so c_↓|↑↓> = -|↑>)
// --------------------------------------------------------------
struct SiteOps
{
    SpMat c_up, c_dn; // annihilation
    SpMat n_up, n_dn; // number
    SpMat Nf;         // total number
    SpMat P;          // (-1)^Nf
    SpMat I4;         // identity 4x4
    // MZ per local basis state: [0, +1, -1, 0]
    std::array<int, 4> MZ = {0, +1, -1, 0};
    // Q per local basis state: [0, 1, 1, 2]
    std::array<int, 4> Q = {0, 1, 1, 2};
};

static SiteOps make_site_ops()
{
    auto sp = [](int r, int c, double v)
    { return Triplet(r, c, v); };
    SpMat I4(4, 4);
    I4.setIdentity();

    // c_up
    std::vector<Triplet> Tcup;
    Tcup.reserve(2);
    Tcup.emplace_back(0, 1, 1.0); // |0><↑|
    Tcup.emplace_back(2, 3, 1.0); // |↓><↑↓|
    SpMat c_up(4, 4);
    c_up.setFromTriplets(Tcup.begin(), Tcup.end());

    // c_dn (note the sign on |↑↓> -> |↑|)
    std::vector<Triplet> Tcdn;
    Tcdn.reserve(2);
    Tcdn.emplace_back(0, 2, 1.0);  // |0><↓|
    Tcdn.emplace_back(1, 3, -1.0); // -|↑><↑↓|
    SpMat c_dn(4, 4);
    c_dn.setFromTriplets(Tcdn.begin(), Tcdn.end());

    // n_up, n_dn, Nf
    SpMat n_up(4, 4), n_dn(4, 4), Nf(4, 4);
    std::vector<Triplet> Tu, Td, Tn;
    Tu.emplace_back(1, 1, 1.0);
    Tu.emplace_back(3, 3, 1.0);
    Td.emplace_back(2, 2, 1.0);
    Td.emplace_back(3, 3, 1.0);
    Tn.emplace_back(1, 1, 1.0);
    Tn.emplace_back(2, 2, 1.0);
    Tn.emplace_back(3, 3, 2.0);
    n_up.setFromTriplets(Tu.begin(), Tu.end());
    n_dn.setFromTriplets(Td.begin(), Td.end());
    Nf.setFromTriplets(Tn.begin(), Tn.end());

    // parity P = (-1)^Nf = diag(1,-1,-1, +1)
    SpMat P(4, 4);
    std::vector<Triplet> Tp;
    Tp.emplace_back(0, 0, 1.0);
    Tp.emplace_back(1, 1, -1.0);
    Tp.emplace_back(2, 2, -1.0);
    Tp.emplace_back(3, 3, 1.0);
    P.setFromTriplets(Tp.begin(), Tp.end());

    return SiteOps{c_up, c_dn, n_up, n_dn, Nf, P, I4};
}

static const SiteOps SITE = make_site_ops();

// --------------------------------------------------------------
// Impurity Hamiltonian H_imp = ed*(n_up+n_dn) + U*n_up*n_dn
// --------------------------------------------------------------
static SpMat impurity_hamiltonian(double U, double ed)
{
    SpMat H(4, 4);
    std::vector<Triplet> T;
    T.reserve(3);
    T.emplace_back(1, 1, ed);
    T.emplace_back(2, 2, ed);
    T.emplace_back(3, 3, 2.0 * ed + U);
    H.setFromTriplets(T.begin(), T.end());
    return H;
}

// --------------------------------------------------------------
// Wilson chain hoppings
// --------------------------------------------------------------
static std::vector<double> wilson_hoppings(double Lambda, int Nch, double D)
{
    std::vector<double> t(std::max(0, Nch - 1));
    for (size_t n = 0; n < t.size(); ++n)
    {
        t[n] = D * (1.0 + 1.0 / Lambda) / 2.0 * std::pow(Lambda, -0.5 * double(n));
    }
    return t;
}

// --------------------------------------------------------------
// Block container (kept basis)
// --------------------------------------------------------------
struct Block
{
    SpMat H;
    SpMat Fup, Fdn;
    SpMat Nup_imp, Ndn_imp;
    SpMat Clast_up, Clast_dn;
    SpMat P;
    std::vector<int> Q, MZ;
};

struct IterData
{
    Mat U_eig;
    std::vector<double> evals;

    Mat U_sorted;
    std::vector<double> evals_sorted;
    int K = 0;
    int K_prev = 0;

    std::vector<int> keep_idx;
    std::vector<int> disc_idx;
    std::vector<int> disc_sorted;

    std::vector<int> Q_cols;
    std::vector<int> MZ_cols;

    std::vector<double> up_abs2_mg, up_abs2_gm;
    std::vector<double> dn_abs2_mg, dn_abs2_gm;

    Mat Fprev_up, Fprev_dn;
    Mat Nprev_up, Nprev_dn;
};

struct NRGRun
{
    std::vector<Block> blocks;
    std::vector<IterData> iters;
};

// --------------------------------------------------------------
// Enlarge quantum numbers by a new site (cartesian sum)
// --------------------------------------------------------------
static void enlarge_QM(const std::vector<int> &Q_old, const std::vector<int> &MZ_old,
                       std::vector<int> &Q_enl, std::vector<int> &MZ_enl)
{
    const int d_old = (int)Q_old.size();
    Q_enl.resize(d_old * 4);
    MZ_enl.resize(d_old * 4);
    for (int i = 0; i < d_old; ++i)
    {
        for (int s = 0; s < 4; ++s)
        {
            const int j = 4 * i + s;
            Q_enl[j] = Q_old[i] + SITE.Q[s];
            MZ_enl[j] = MZ_old[i] + SITE.MZ[s];
        }
    }
}

struct SectorRange
{
    int q, mz;
    int start, end;
};

static std::pair<std::vector<int>, std::vector<SectorRange>>
sector_permutation(const std::vector<int> &Q, const std::vector<int> &MZ)
{
    const int n = (int)Q.size();
    std::map<std::pair<int, int>, std::vector<int>> buckets;
    for (int i = 0; i < n; ++i)
        buckets[{Q[i], MZ[i]}].push_back(i);

    std::vector<int> perm;
    perm.reserve(n);
    std::vector<SectorRange> ranges;
    ranges.reserve(buckets.size());
    int cursor = 0;
    for (auto &kv : buckets)
    {
        auto &lst = kv.second;
        SectorRange R{kv.first.first, kv.first.second, cursor, cursor + (int)lst.size()};
        ranges.push_back(R);
        for (int idx : lst)
            perm.push_back(idx);
        cursor += (int)lst.size();
    }
    return {perm, ranges};
}

static Mat permute_dense(const Mat &A, const std::vector<int> &perm)
{
    const int n = (int)perm.size();
    Mat B(n, n);
    B.setZero();
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            B(i, j) = A(perm[i], perm[j]);
        }
    }
    return B;
}

static std::vector<int> inv_perm(const std::vector<int> &perm)
{
    std::vector<int> inv(perm.size());
    for (size_t i = 0; i < perm.size(); ++i)
        inv[perm[i]] = (int)i;
    return inv;
}

static void block_eigen(const SpMat &H_enl,
                        const std::vector<int> &Q_enl,
                        const std::vector<int> &MZ_enl,
                        std::vector<double> &evals_full,
                        Mat &U_full,
                        std::vector<int> &col_Q,
                        std::vector<int> &col_MZ)
{
    const int n = H_enl.rows();
    assert(H_enl.cols() == n);

    auto [perm, ranges] = sector_permutation(Q_enl, MZ_enl);
    Mat Hperm = permute_dense(dense(H_enl), perm);

    Mat Uperm = Mat::Zero(n, n);
    evals_full.resize(n);
    col_Q.resize(n);
    col_MZ.resize(n);

    for (const auto &R : ranges)
    {
        const int s = R.start, dim = R.end - R.start;
        if (dim <= 0)
            continue;
        Mat block = Hperm.block(s, s, dim, dim);
        Eigen::SelfAdjointEigenSolver<Mat> es(block);
        Mat Us = es.eigenvectors();
        Vec Es = es.eigenvalues();
        Uperm.block(s, s, dim, dim) = Us;
        for (int i = 0; i < dim; ++i)
        {
            evals_full[s + i] = Es(i);
            col_Q[s + i] = R.q;
            col_MZ[s + i] = R.mz;
        }
    }

    auto invp = inv_perm(perm);
    U_full = Mat::Zero(n, n);
    for (int i = 0; i < n; ++i)
    {
        U_full.row(i) = Uperm.row(invp[i]);
    }
}

// --------------------------------------------------------------
// Project operator A_enl to the kept subspace given eigenvectors (sorted)
// パフォーマンス改善：A_enl を密行列にせず、疎行列積のまま計算
// --------------------------------------------------------------
static SpMat project_op_keep(const SpMat &A_enl, const Mat &U_sorted, int K)
{
    Mat Aeig = U_sorted.transpose() * (A_enl * U_sorted); // Sparse * Dense -> Dense
    Mat Akeep = Aeig.block(0, 0, K, K);
    return sparse(Akeep);
}

// --------------------------------------------------------------
// Initialize the impurity-only block (size 4)
// --------------------------------------------------------------
static Block init_block(double U, double ed)
{
    Block B;
    B.H = impurity_hamiltonian(U, ed);
    B.Fup = SITE.c_up;
    B.Fdn = SITE.c_dn;
    B.Nup_imp = SITE.n_up;
    B.Ndn_imp = SITE.n_dn;
    B.Clast_up = SpMat(4, 4);
    B.Clast_dn = SpMat(4, 4);
    B.P = SITE.P;
    B.Q = {0, 1, 1, 2};
    B.MZ = {0, 1, -1, 0};
    return B;
}

// ---------- Enlarge + diagonalize + truncate ----------
static void add_site_and_truncate(Block &block, IterData &it,
                                  const std::string &couple_kind,
                                  double V_or_t, int NS, bool full_keep = false)
{
    const int d_old = block.H.rows();

    it.K_prev = d_old;
    it.Fprev_up = dense(block.Fup);
    it.Fprev_dn = dense(block.Fdn);
    it.Nprev_up = dense(block.Nup_imp);
    it.Nprev_dn = dense(block.Ndn_imp);

    const SpMat &I4 = SITE.I4;

    SpMat H_enl = kron_sp(block.H, I4);
    SpMat Fup_enl = kron_sp(block.Fup, I4);
    SpMat Fdn_enl = kron_sp(block.Fdn, I4);
    SpMat Nup_enl = kron_sp(block.Nup_imp, I4);
    SpMat Ndn_enl = kron_sp(block.Ndn_imp, I4);

    SpMat Cup_new_global = kron_sp(block.P, SITE.c_up);
    SpMat Cdn_new_global = kron_sp(block.P, SITE.c_dn);

    if (couple_kind == "impurity")
    {
        H_enl += V_or_t * (Fup_enl.transpose() * Cup_new_global + Cup_new_global.transpose() * Fup_enl);
        H_enl += V_or_t * (Fdn_enl.transpose() * Cdn_new_global + Cdn_new_global.transpose() * Fdn_enl);
    }
    else if (couple_kind == "chain")
    {
        if (block.Clast_up.rows() > 0)
        {
            SpMat Cup_prev = kron_sp(block.Clast_up, SITE.I4);
            SpMat Cdn_prev = kron_sp(block.Clast_dn, SITE.I4);
            H_enl += V_or_t * (Cup_prev.transpose() * Cup_new_global + Cup_new_global.transpose() * Cup_prev);
            H_enl += V_or_t * (Cdn_prev.transpose() * Cdn_new_global + Cdn_new_global.transpose() * Cdn_prev);
        }
        else
        {
            H_enl += V_or_t * (Fup_enl.transpose() * Cup_new_global + Cup_new_global.transpose() * Fup_enl);
            H_enl += V_or_t * (Fdn_enl.transpose() * Cdn_new_global + Cdn_new_global.transpose() * Fdn_enl);
        }
    }
    else
    {
        throw std::runtime_error("Unknown couple_kind");
    }

    std::vector<int> Q_enl, MZ_enl;
    enlarge_QM(block.Q, block.MZ, Q_enl, MZ_enl);

    std::vector<double> evals_full;
    Mat U_full;
    std::vector<int> col_Q, col_MZ;
    block_eigen(H_enl, Q_enl, MZ_enl, evals_full, U_full, col_Q, col_MZ);

    std::vector<int> idx(evals_full.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b)
                     { return evals_full[a] < evals_full[b]; });

    const int n = (int)idx.size();
    const int K = full_keep ? n : std::min(NS, n);

    Mat U_sorted(n, n);
    for (int j = 0; j < n; ++j)
        U_sorted.col(j) = U_full.col(idx[j]);

    std::vector<double> evals_sorted(n);
    for (int j = 0; j < n; ++j)
        evals_sorted[j] = evals_full[idx[j]];

    SpMat H_keep = sparse(Mat::Zero(K, K));
    {
        Vec evals_vec(n);
        for (int j = 0; j < n; ++j)
            evals_vec(j) = evals_sorted[j];
        H_keep = sparse(evals_vec.head(K).asDiagonal());
    }

    SpMat Fup_keep = project_op_keep(Fup_enl, U_sorted, K);
    SpMat Fdn_keep = project_op_keep(Fdn_enl, U_sorted, K);
    SpMat Nup_keep = project_op_keep(Nup_enl, U_sorted, K);
    SpMat Ndn_keep = project_op_keep(Ndn_enl, U_sorted, K);

    SpMat Clast_up_keep = project_op_keep(Cup_new_global, U_sorted, K);
    SpMat Clast_dn_keep = project_op_keep(Cdn_new_global, U_sorted, K);

    SpMat P_enl = kron_sp(block.P, SITE.P);
    SpMat P_keep = project_op_keep(P_enl, U_sorted, K);

    std::vector<int> Q_keep(K), MZ_keep(K);
    for (int j = 0; j < K; ++j)
    {
        Q_keep[j] = col_Q[idx[j]];
        MZ_keep[j] = col_MZ[idx[j]];
    }

    const int g = 0;

    // パフォーマンス改善：疎行列のまま積を計算
    Mat Fup_eig = U_sorted.transpose() * (Fup_enl * U_sorted);
    Mat Fdn_eig = U_sorted.transpose() * (Fdn_enl * U_sorted);

    it.up_abs2_mg.clear();
    it.up_abs2_gm.clear();
    it.dn_abs2_mg.clear();
    it.dn_abs2_gm.clear();
    it.disc_sorted.clear();

    for (int m = K; m < n; ++m)
    {
        const double a_mg_up = Fup_eig(m, g);
        const double a_gm_up = Fup_eig(g, m);
        const double a_mg_dn = Fdn_eig(m, g);
        const double a_gm_dn = Fdn_eig(g, m);
        it.up_abs2_mg.push_back(a_mg_up * a_mg_up);
        it.up_abs2_gm.push_back(a_gm_up * a_gm_up);
        it.dn_abs2_mg.push_back(a_mg_dn * a_mg_dn);
        it.dn_abs2_gm.push_back(a_gm_dn * a_gm_dn);
        it.disc_sorted.push_back(m);
    }

    it.U_eig = U_full;
    it.evals = evals_full;
    it.U_sorted = U_sorted;
    it.evals_sorted = std::move(evals_sorted);
    it.K = K;
    it.keep_idx.assign(idx.begin(), idx.begin() + K);
    it.disc_idx.assign(idx.begin() + K, idx.end());

    it.Q_cols = col_Q;
    it.MZ_cols = col_MZ;

    block.H = std::move(H_keep);
    block.Fup = std::move(Fup_keep);
    block.Fdn = std::move(Fdn_keep);
    block.Nup_imp = std::move(Nup_keep);
    block.Ndn_imp = std::move(Ndn_keep);
    block.Clast_up = std::move(Clast_up_keep);
    block.Clast_dn = std::move(Clast_dn_keep);
    block.P = std::move(P_keep);
    block.Q = std::move(Q_keep);
    block.MZ = std::move(MZ_keep);
}

// --------------------------------------------------------------
// Run NRG with list of hoppings
// --------------------------------------------------------------
static NRGRun run_NRG_with_tlist(double U, double ed, double V,
                                 const std::vector<double> &tlist,
                                 int NS)
{
    NRGRun run;
    run.blocks.clear();
    run.iters.clear();

    Block B = init_block(U, ed);
    run.blocks.push_back(B);

    {
        IterData it;
        add_site_and_truncate(B, it, "impurity", V, NS, /*full_keep=*/false);
        run.iters.push_back(std::move(it));
        run.blocks.push_back(B);
    }

    for (size_t n = 0; n < tlist.size(); ++n)
    {
        IterData it;
        add_site_and_truncate(B, it, "chain", tlist[n], NS, /*full_keep=*/false);
        run.iters.push_back(std::move(it));
        run.blocks.push_back(B);
    }

    return run;
}

// ---------- Run NRG (return final kept block) ----------
static NRGRun run_NRG(double U, double ed, double V, double Lambda, int Nch, double D, int NS)
{
    auto tlist = wilson_hoppings(Lambda, Nch, D);
    return run_NRG_with_tlist(U, ed, V, tlist, NS);
}

struct DiscreteSpectrum
{
    std::vector<double> w;
    std::vector<double> weight;
};

static Mat partial_trace_right(const Mat &R, int dim_left, int dim_right)
{
    Mat rho = Mat::Zero(dim_left, dim_left);
    for (int a = 0; a < dim_left; ++a)
    {
        for (int b = 0; b < dim_left; ++b)
        {
            double s = 0.0;
            for (int sidx = 0; sidx < dim_right; ++sidx)
            {
                int i = a * dim_right + sidx;
                int j = b * dim_right + sidx;
                s += R(i, j);
            }
            rho(a, b) = s;
        }
    }
    return rho;
}

static std::vector<Mat> propagate_rdm_backward(const NRGRun &run)
{
    const int L = (int)run.iters.size();
    std::vector<Mat> rhos(L);

    const auto &Ilast = run.iters.back();
    const int Klast = Ilast.K;
    Mat rho = Mat::Zero(Klast, Klast);
    const double Eg = Ilast.evals_sorted[0];
    const double tol = 1e-12;
    std::vector<int> gs;
    for (int k = 0; k < Klast; ++k)
    {
        if (std::abs(Ilast.evals_sorted[k] - Eg) < tol)
            gs.push_back(k);
        else
            break;
    }
    const double w = 1.0 / std::max(1, (int)gs.size());
    for (int g : gs)
        rho(g, g) += w;
    rhos[L - 1] = rho;

    for (int i = L - 1; i > 0; --i)
    {
        const auto &I = run.iters[i];
        const int K = I.K;
        const int dsite = 4;
        const int Kprev = I.K_prev;

        Mat U_K = I.U_sorted.leftCols(K);
        Mat rho_enl = U_K * rho * U_K.transpose();
        double trE = rho_enl.trace();
        if (std::abs(trE - 1.0) > 1e-12)
            rho_enl /= trE;

        Mat rho_prev = partial_trace_right(rho_enl, Kprev, dsite);
        double tr = rho_prev.trace();
        if (std::abs(tr - 1.0) > 1e-12)
            rho_prev /= tr;

        rhos[i - 1] = rho_prev;
        rho.swap(rho_prev);
    }
    return rhos;
}

static DiscreteSpectrum cfs_discrete_T0_rdm(
    const NRGRun &run, const std::string &spin, const std::vector<Mat> &rhos)
{
    DiscreteSpectrum ds;
    ds.w.reserve(1 << 14);
    ds.weight.reserve(1 << 14);
    const int L = (int)run.iters.size();

    for (int i = 0; i < L; ++i)
    {
        const auto &I = run.iters[i];
        if (I.evals_sorted.empty())
            continue;
        const int n = (int)I.U_sorted.rows(), K = I.K;
        if (K <= 0 || K > n)
            continue;
        const Mat &rhoK = rhos[i];
        if (rhoK.rows() != K || rhoK.cols() != K)
            continue;

        const double Eg = I.evals_sorted[0];

        const Mat Fprev = (spin == "up" ? I.Fprev_up : I.Fprev_dn);
        const SpMat F_enl = kron_sp(sparse(Fprev), SITE.I4);

        // パフォーマンス改善：疎行列積
        const Mat F_eig = I.U_sorted.transpose() * (F_enl * I.U_sorted);

        const int Ksum = (i == L - 1 ? 0 : K);

        for (int m = Ksum; m < n; ++m)
        {
            const double w = I.evals_sorted[m] - Eg;
            if (w <= 1e-14)
                continue;

            Eigen::RowVectorXd r = F_eig.row(m).leftCols(K);
            Eigen::VectorXd c = F_eig.block(0, m, K, 1);

            double wpos = (r * rhoK * r.transpose())(0, 0);
            double wneg = (c.transpose() * rhoK * c)(0, 0);

            if (wpos > 0)
            {
                ds.w.push_back(+w);
                ds.weight.push_back(wpos);
            }
            if (wneg > 0)
            {
                ds.w.push_back(-w);
                ds.weight.push_back(wneg);
            }
        }
    }

    double tot = 0;
    for (double v : ds.weight)
        tot += v;
    std::cerr << "# [CFS-RDM] sum of weights = " << std::setprecision(12) << tot << "\n";
    return ds;
}

static DiscreteSpectrum cfs_discrete_T0_rdm_pair_conn(
    const NRGRun &run, const std::string &spin,
    const std::vector<Mat> &rhos, double nbar_fix)
{
    DiscreteSpectrum ds;
    ds.w.reserve(1 << 14);
    ds.weight.reserve(1 << 14);
    const int L = (int)run.iters.size();

    for (int i = 0; i < L; ++i)
    {
        const auto &I = run.iters[i];
        if (I.evals_sorted.empty())
            continue;
        const int n = (int)I.U_sorted.rows(), K = I.K;
        if (K <= 0 || K > n)
            continue;
        const Mat &rhoK = rhos[i];
        if (rhoK.rows() != K || rhoK.cols() != K)
            continue;

        const double Eg = I.evals_sorted[0];

        const Mat Fprev = (spin == "up" ? I.Fprev_up : I.Fprev_dn);
        const Mat Nprev = (spin == "up" ? I.Nprev_dn : I.Nprev_up);
        const int Kprev = I.K_prev;
        if (Kprev <= 0)
            continue;
        assert(Fprev.rows() == Kprev && Fprev.cols() == Kprev);
        assert(Nprev.rows() == Kprev && Nprev.cols() == Kprev);

        const Mat Aprev = (Nprev - nbar_fix * Mat::Identity(Kprev, Kprev)) * Fprev;
        const SpMat A_enl = kron_sp(sparse(Aprev), SITE.I4);
        const SpMat B_enl = kron_sp(sparse(Fprev), SITE.I4);

        // パフォーマンス改善：疎行列積
        const Mat A_eig = I.U_sorted.transpose() * (A_enl * I.U_sorted);
        const Mat B_eig = I.U_sorted.transpose() * (B_enl * I.U_sorted);

        const int Ksum = (i == L - 1 ? 0 : K);

        for (int m = Ksum; m < n; ++m)
        {
            const double w = I.evals_sorted[m] - Eg;
            if (w <= 1e-14)
                continue;

            Eigen::RowVectorXd rA = A_eig.row(m).leftCols(K);
            Eigen::RowVectorXd rB = B_eig.row(m).leftCols(K);
            Eigen::VectorXd cA = A_eig.block(0, m, K, 1);
            Eigen::VectorXd cB = B_eig.block(0, m, K, 1);

            double Wpos = (cB.transpose() * rhoK * cA)(0, 0);
            double Wneg = (rA * rhoK * rB.transpose())(0, 0);

            if (std::abs(Wpos) > 1e-18)
            {
                ds.w.push_back(+w);
                ds.weight.push_back(Wpos);
            }
            if (std::abs(Wneg) > 1e-18)
            {
                ds.w.push_back(-w);
                ds.weight.push_back(Wneg);
            }
        }
    }

    auto soft_dc_fix = [&](DiscreteSpectrum &d)
    {
        double S = 0;
        for (double v : d.weight)
            S += v;
        if (std::abs(S) <= 1e-12)
            return;
        int ip = -1, in = -1;
        double wp = 1e300, wn = 1e300;
        for (int k = 0; k < (int)d.w.size(); ++k)
        {
            double x = d.w[k];
            if (x > 0 && x < wp)
            {
                wp = x;
                ip = k;
            }
            if (x < 0 && std::abs(x) < wn)
            {
                wn = std::abs(x);
                in = k;
            }
        }
        if (ip >= 0 && in >= 0)
        {
            d.weight[ip] -= 0.5 * S;
            d.weight[in] -= 0.5 * S;
        }
    };
    soft_dc_fix(ds);

    double tot = 0;
    for (double v : ds.weight)
        tot += v;
    std::cerr << "# [CFS-RDM-PAIR-CONN] sum of weights = " << std::setprecision(12) << tot << "\n";
    return ds;
}

// ---- Hilbert 変換（実軸の主値積分、等間隔グリッド前提）----
// 精度改善：区分的線形補間を用いた解析的な主値積分の計算
static std::vector<double> kk_hilbert(const std::vector<double> &w,
                                      const std::vector<double> &A)
{
    const int N = (int)w.size();
    std::vector<double> Re(N, 0.0);
    if (N < 2)
        return Re;

    for (int i = 0; i < N; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < N - 1; ++j)
        {
            double x0 = w[j];
            double x1 = w[j + 1];
            double y0 = A[j];
            double y1 = A[j + 1];
            double slope = (y1 - y0) / (x1 - x0);

            // 積分領域の定数項
            s -= slope * (x1 - x0);

            // 等間隔グリッドでの極（i == j および i == j+1）付近では、
            // 対数発散項が左右の区間で完全に相殺されるため明示的にスキップする
            if (i != j && i != j + 1)
            {
                double d0 = std::abs(w[i] - x0);
                double d1 = std::abs(w[i] - x1);
                double y_extrap = y0 + slope * (w[i] - x0);
                s += y_extrap * std::log(d0 / d1);
            }
        }
        Re[i] = s;
    }
    return Re;
}

static DiscreteSpectrum cfs_discrete_T0_rdm_op(
    const NRGRun &run, const std::string &spin,
    const std::vector<Mat> &rhos, bool numerator)
{
    DiscreteSpectrum ds;
    ds.w.reserve(1 << 14);
    ds.weight.reserve(1 << 14);
    const int L = (int)run.iters.size();
    for (int i = 0; i < L; ++i)
    {
        const auto &I = run.iters[i];
        if (I.evals_sorted.empty())
            continue;
        const int n = (int)I.U_sorted.rows(), K = I.K;
        if (K <= 0 || K > n)
            continue;
        const Mat &rhoK = rhos[i];
        if (rhoK.rows() != K || rhoK.cols() != K)
            continue;
        const double Eg = I.evals_sorted[0];

        const Mat Fprev = (spin == "up" ? I.Fprev_up : I.Fprev_dn);
        Mat Oprev = Fprev;
        if (numerator)
        {
            const Mat Nprev = (spin == "up" ? I.Nprev_dn : I.Nprev_up);
            Oprev = Nprev * Fprev;
        }
        const SpMat O_enl = kron_sp(sparse(Oprev), SITE.I4);

        // パフォーマンス改善：疎行列積
        const Mat O_eig = I.U_sorted.transpose() * (O_enl * I.U_sorted);

        const int Ksum = (i == L - 1 ? 0 : K);
        for (int m = Ksum; m < n; ++m)
        {
            const double w = I.evals_sorted[m] - Eg;
            if (w <= 1e-14)
                continue;
            Eigen::RowVectorXd r = O_eig.row(m).leftCols(K);
            Eigen::VectorXd c = O_eig.block(0, m, K, 1);
            double Wpos = (c.transpose() * rhoK * c)(0, 0);
            double Wneg = (r * rhoK * r.transpose())(0, 0);
            if (Wpos > 0)
            {
                ds.w.push_back(+w);
                ds.weight.push_back(Wpos);
            }
            if (Wneg > 0)
            {
                ds.w.push_back(-w);
                ds.weight.push_back(Wneg);
            }
        }
    }
    return ds;
}

static double nbar_final_from_rdm(const NRGRun &run,
                                  const std::vector<Mat> &rhos,
                                  const std::string &spin)
{
    const Mat &rhoK = rhos.back();
    const auto &Bfin = run.blocks.back();
    Mat Nk = dense((spin == "up") ? Bfin.Ndn_imp
                                  : Bfin.Nup_imp);
    assert(rhoK.rows() == Nk.rows() && rhoK.cols() == Nk.cols());
    return (rhoK * Nk).trace();
}

// ---------- log-Gaussian smoothing (mass-preserving per spike) ----------
static std::vector<double> linspace(double a, double b, int n)
{
    std::vector<double> x(n);
    if (n == 1)
    {
        x[0] = a;
        return x;
    }
    double h = (b - a) / double(n - 1);
    for (int i = 0; i < n; ++i)
        x[i] = a + h * i;
    return x;
}

static std::vector<double>
A_from_spikes_log_gaussian(const DiscreteSpectrum &ds,
                           const std::vector<double> &wgrid,
                           double b_log)
{
    const double PI = 3.14159265358979323846;
    const double inv_s2 = 0.5 / (b_log * b_log);
    const double norm = 1.0 / (b_log * std::sqrt(2.0 * PI));
    std::vector<double> A(wgrid.size(), 0.0);
    if (wgrid.size() < 2)
        return A;
    const double dw = wgrid[1] - wgrid[0];

    std::vector<double> Ki(wgrid.size());
    for (size_t k = 0; k < ds.w.size(); ++k)
    {
        const double w0 = ds.w[k], wk = ds.weight[k];
        if (std::abs(w0) < 1e-14 || wk == 0.0)
            continue;

        double denom = 0.0;
        for (size_t i = 0; i < wgrid.size(); ++i)
        {
            const double w = wgrid[i];
            if (w * w0 <= 0)
            {
                Ki[i] = 0.0;
                continue;
            }
            const double y = std::log(std::abs(w / w0));
            const double K = norm * std::exp(-y * y * inv_s2) / std::abs(w);
            Ki[i] = K;
            denom += K * dw;
        }
        const double scale = (denom > 0.0 ? 1.0 / denom : 1.0);
        for (size_t i = 0; i < wgrid.size(); ++i)
            if (Ki[i] != 0.0)
                A[i] += wk * Ki[i] * scale;
    }
    return A;
}

static std::vector<double>
A_from_spikes_hybrid(const DiscreteSpectrum &ds,
                     const std::vector<double> &wgrid,
                     double b_log,
                     double w_lin,
                     double sigma_lin)
{
    const double PI = 3.14159265358979323846;

    if (wgrid.empty() || !(b_log > 0.0))
        return std::vector<double>(wgrid.size(), 0.0);
    if (sigma_lin <= 0.0)
        sigma_lin = 0.2 * std::max(1e-12, w_lin);

    const double inv_s2_log = 0.5 / (b_log * b_log);
    const double norm_log = 1.0 / (b_log * std::sqrt(2.0 * PI));
    const double norm_lin = 1.0 / (sigma_lin * std::sqrt(2.0 * PI));

    std::vector<double> A(wgrid.size(), 0.0);
    if (wgrid.size() < 2)
        return A;
    const double dw = wgrid[1] - wgrid[0];

    std::vector<double> Ki(wgrid.size());

    for (size_t k = 0; k < ds.w.size(); ++k)
    {
        const double w0 = ds.w[k];
        const double wk = ds.weight[k];
        if (!std::isfinite(w0) || !std::isfinite(wk) || wk == 0.0)
            continue;
        if (std::abs(w0) < 1e-15)
            continue;

        double denom = 0.0;

        for (size_t i = 0; i < wgrid.size(); ++i)
        {
            const double w = wgrid[i];
            double K = 0.0;

            if (std::abs(w) < w_lin)
            {
                const double z = (w - w0) / sigma_lin;
                K = norm_lin * std::exp(-0.5 * z * z);
            }
            else
            {
                if (w * w0 > 0.0)
                {
                    const double absw = std::abs(w);
                    if (absw > 0.0)
                    {
                        const double y = std::log(std::abs(w / w0));
                        K = norm_log * std::exp(-y * y * inv_s2_log) / absw;
                    }
                }
            }

            if (!std::isfinite(K))
                K = 0.0;
            Ki[i] = K;

            const double add = K * dw;
            if (std::isfinite(add))
                denom += add;
        }

        double scale = (denom > 0.0 ? 1.0 / denom : 0.0);
        if (!std::isfinite(scale))
            scale = 0.0;
        const double wksc = wk * scale;

        for (size_t i = 0; i < wgrid.size(); ++i)
        {
            if (Ki[i] != 0.0)
                A[i] += wksc * Ki[i];
        }
    }

    for (double &v : A)
        if (!std::isfinite(v))
            v = 0.0;
    return A;
}

// ---------- helpers ----------
static double trapz(const std::vector<double> &x, const std::vector<double> &y)
{
    double S = 0;
    for (size_t i = 1; i < x.size(); ++i)
        S += 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1]);
    return S;
}

static double interp0(const std::vector<double> &w, const std::vector<double> &A)
{
    int ip = -1, in = -1;
    for (size_t i = 0; i < w.size(); ++i)
        if (w[i] > 0)
        {
            ip = (int)i;
            break;
        }
    for (int i = (int)w.size() - 1; i >= 0; --i)
        if (w[i] < 0)
        {
            in = i;
            break;
        }
    if (ip < 0 || in < 0)
    {
        size_t k = std::min_element(w.begin(), w.end(),
                                    [](double a, double b)
                                    { return std::abs(a) < std::abs(b); }) -
                   w.begin();
        return A[k];
    }
    double x0 = w[in], y0 = A[in], x1 = w[ip], y1 = A[ip];
    return y0 + (y1 - y0) * (0 - x0) / (x1 - x0);
}

static double n_from_A(const std::vector<double> &w, const std::vector<double> &A)
{
    if (w.empty() || w.size() != A.size())
        return 0.0;

    std::vector<double> wn, An;
    wn.reserve(w.size() + 1);
    An.reserve(A.size() + 1);

    bool has_zero = false;
    int iz = -1;
    for (size_t i = 0; i < w.size(); ++i)
    {
        if (std::abs(w[i]) < 1e-15)
        {
            has_zero = true;
            iz = (int)i;
        }
        if (w[i] < 0.0)
        {
            wn.push_back(w[i]);
            An.push_back(std::max(0.0, A[i]));
        }
    }

    // スコープと安全性の改善
    double A0 = 0.0;
    if (has_zero && iz >= 0 && iz < (int)A.size())
    {
        A0 = std::max(0.0, A[iz]);
    }
    else
    {
        A0 = std::max(0.0, interp0(w, A));
    }

    wn.push_back(0.0);
    An.push_back(A0);

    if (wn.size() < 2)
        return 0.0;
    return trapz(wn, An);
}

// ---------- Args ----------
struct Args
{
    double U = 1.0, ed = -0.5 * U, V = 0.25, Lambda = 2.5, D = 1.0;
    int Nch = 50, NS = 100;
    double wmin = -1.5, wmax = 1.5;
    int ngrid = 15001;
    double blog = 0.5;
    std::string spin = "up";
    double w_lin = 1e-2;
    double sigma_lin = 0.0;
    std::string out = "out";
};

static Args parse_args(int argc, char **argv)
{
    Args a;
    auto next = [&](int &i)
    { return std::string(argv[++i]); };
    for (int i = 1; i < argc; ++i)
    {
        std::string s = argv[i];
        if (s == "--U")
            a.U = std::stod(next(i));
        else if (s == "--ed")
            a.ed = std::stod(next(i));
        else if (s == "--V")
            a.V = std::stod(next(i));
        else if (s == "--Lambda")
            a.Lambda = std::stod(next(i));
        else if (s == "--Nch")
            a.Nch = std::stoi(next(i));
        else if (s == "--NS")
            a.NS = std::stoi(next(i));
        else if (s == "--D")
            a.D = std::stod(next(i));
        else if (s == "--wmin")
            a.wmin = std::stod(next(i));
        else if (s == "--wmax")
            a.wmax = std::stod(next(i));
        else if (s == "--ngrid")
            a.ngrid = std::stoi(next(i));
        else if (s == "--blog")
            a.blog = std::stod(next(i));
        else if (s == "--spin")
            a.spin = next(i);
        else if (s == "--wlin")
            a.w_lin = std::stod(next(i));
        else if (s == "--sigmalin")
            a.sigma_lin = std::stod(next(i));
        else if (s == "--out")
            a.out = next(i);
        else
            std::cerr << "Unknown arg: " << s << "\n";
    }
    return a;
}

// ---------- main ----------
int main(int argc, char **argv)
{
    Args A = parse_args(argc, argv);

    NRGRun run = run_NRG(A.U, A.ed, A.V, A.Lambda, A.Nch, A.D, A.NS);

    auto rhos = propagate_rdm_backward(run);

    double ns = nbar_final_from_rdm(run, rhos, A.spin);
    DiscreteSpectrum dsG = cfs_discrete_T0_rdm_op(run, A.spin, rhos, false);
    DiscreteSpectrum dsF_conn = cfs_discrete_T0_rdm_pair_conn(run, A.spin, rhos, ns);

    std::vector<double> w = linspace(A.wmin, A.wmax, A.ngrid);
    auto smooth = [&](const DiscreteSpectrum &ds)
    {
        return (A.w_lin > 0.0)
                   ? A_from_spikes_hybrid(ds, w, A.blog, A.w_lin, A.sigma_lin)
                   : A_from_spikes_log_gaussian(ds, w, A.blog);
    };

    auto enforce_even = [&](std::vector<double> &y)
    {
        size_t N = y.size();
        for (size_t i = 0; i < N / 2; ++i)
        {
            double v = 0.5 * (y[i] + y[N - 1 - i]);
            y[i] = y[N - 1 - i] = v;
        }
    };
    auto enforce_odd = [&](std::vector<double> &y)
    {
        size_t N = y.size();
        for (size_t i = 0; i < N / 2; ++i)
        {
            double v = 0.5 * (y[i] - y[N - 1 - i]);
            y[i] = v;
            y[N - 1 - i] = -v;
        }
        if (N % 2 == 1)
            y[N / 2] = 0.0;
    };

    std::vector<double> Gspec = smooth(dsG);
    std::vector<double> Fspec = smooth(dsF_conn);

    double IG = trapz(w, Gspec);
    if (IG > 0.0 && std::abs(IG - 1.0) > 1e-10)
        for (auto &v : Gspec)
            v /= IG;

    double IFc = trapz(w, Fspec);
    if (std::abs(IFc) > 1e-12)
        for (size_t i = 0; i < Fspec.size(); ++i)
            Fspec[i] -= IFc * Gspec[i];

    if (std::abs(A.ed + 0.5 * A.U) < 1e-12)
    {
        enforce_even(Gspec);
        enforce_odd(Fspec);
    }

    std::vector<double> Ftot(Fspec.size());
    for (size_t i = 0; i < Ftot.size(); ++i)
        Ftot[i] = Fspec[i] + ns * Gspec[i];

    std::vector<double> ImG(Gspec.size()), ImF(Fspec.size());
    for (size_t i = 0; i < Gspec.size(); ++i)
    {
        ImG[i] = -M_PI * Gspec[i];
        ImF[i] = -M_PI * Ftot[i];
    }
    std::vector<double> ReG = kk_hilbert(w, Gspec);
    std::vector<double> ReF = kk_hilbert(w, Ftot);

    const double PI = 3.14159265358979323846;
    const double rho0 = 1.0 / (2.0 * A.D);
    const double Delta = PI * A.V * A.V * rho0;

    size_t i0 = w.size() / 2;
    std::vector<std::complex<double>> Sigma(w.size());
    auto sig_ratio = [&](size_t i)
    {
        std::complex<double> Gc(ReG[i], ImG[i]), Fc(ReF[i], ImF[i]);
        if (std::abs(Gc) < 1e-16)
            return std::complex<double>(NAN, NAN);
        return A.U * (Fc / Gc);
    };
    for (size_t i = 0; i < w.size(); ++i)
        Sigma[i] = sig_ratio(i);

    double U0_turnon = 0.1;
    double wU = std::tanh(std::abs(A.U) / U0_turnon);

    bool ph = (std::abs(A.ed + 0.5 * A.U) < 1e-12);

    double nsG = n_from_A(w, Gspec);
    double nsUse = wU * ns + (1.0 - wU) * nsG;

    // --- (A’) FR ピン止め：ed を微調整して eps_tilde=Δ cot(πn) を満たす
    double eps_tilde = A.ed + std::real(Sigma[i0]);
    double eps_target = Delta * std::cos(PI * nsUse) / std::sin(PI * nsUse);
    double Cpin = eps_tilde - eps_target;
    double ed_eff = A.ed - wU * Cpin;

    // --- (B) ImΣ の FL クランプ（|ω|<w_cut だけ）
    double Cim = -wU * std::imag(Sigma[i0]);
    double w_cut = 3.0 * std::max(1e-3, A.w_lin);
    for (size_t i = 0; i < w.size(); ++i)
        if (std::abs(w[i]) <= w_cut)
            Sigma[i] += std::complex<double>(0.0, Cim);

    std::vector<double> A_ratio(w.size(), 0.0);
    for (size_t i = 0; i < w.size(); ++i)
    {
        std::complex<double> denom = (w[i] - ed_eff) - Sigma[i] + std::complex<double>(0.0, Delta);
        std::complex<double> Gdy = 1.0 / denom;
        A_ratio[i] = -(1.0 / PI) * std::imag(Gdy);
    }

    double A0_ratio = interp0(w, A_ratio);
    const double A_PH = 1.0 / (PI * Delta);
    const double A_FR = std::sin(PI * ns) * std::sin(PI * ns) / (PI * Delta);

    std::cerr << "# [CHK] integral A_G=" << trapz(w, Gspec)
              << "# [CHK] integral F_conn=" << trapz(w, Fspec)
              << "# [CHK] integral F_tot(ns)=" << trapz(w, Ftot)
              << "  ns=" << ns
              << "  A(0)=" << A0_ratio
              << "  A_PH=" << A_PH
              << "  A_FR=" << A_FR << "\n";

    auto S0 = Sigma[i0];
    // ピン止め補正量（Cpin, Cim）の出力を追加
    std::cerr << std::setprecision(12)
              << "# [PIN] ReSigma(0)=" << S0.real() << " ImSigma(0)=" << S0.imag()
              << "  eps_tilde=" << (ed_eff + S0.real()) << "  Gamma_eff=" << (Delta - S0.imag()) << "\n"
              << "# [PIN] Correction Cpin=" << Cpin << "  Cim=" << Cim << "\n";

    save_csv_two(A.out + "/omega_A.csv", w, A_ratio, "omega", "A");

    std::cout << "Done. Wrote " << (A.out + "/omega_A.csv") << std::endl;
    return 0;
}